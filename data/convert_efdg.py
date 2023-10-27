from typing import Optional, List, Iterator
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin, dataclass_json, Undefined, CatchAll

from hudl_breakdowns_core import Moment
from hudl_moment_datasets import VideoRecord, MomentDataset
from dataclasses import dataclass
from datetime import timedelta
PERIOD_TAG = "period_num"
from pathlib import Path
from hudl_grail.ops import time_concat

import boto3
from typing import List, Optional
from hudl_grail import Chunk
import pickle
from tqdm import tqdm
import gzip
import json
import numpy as np
from fire import Fire


@dataclass
class Period(DataClassJsonMixin):
    number: int
    event_kickoff_ms: float
    """Timestamp of the kickoff of this period
    relative to the start of the broadcast video
    """
    tracking_kickoff_frame: int
    """Frame # of the kickoff of this period
    in the raw video and in the original tracking data
    (not the trimmed stream linked on the EFDGVideoRecord)
    """

    @property
    def event_kickoff(self) -> timedelta:
        """Timestamp of the kickoff of this period
        relative to the start of the broadcast video
        """
        return timedelta(milliseconds=self.event_kickoff_ms)

def adjust_moments_against_periods(
    moments: List[Moment], periods: List[Period]
) -> Iterator[Moment]:
    """Adjust the timestamps on these moments to all be relative to the chunk stream
    Out of the dataset, they come in as relative to the period they're part of.
    """
    period_offsets = {p.number: int(p.event_kickoff_ms) for p in periods}
    for moment in moments:
        period_num = moment.get_tag_value(PERIOD_TAG)
        assert (
            period_num is not None
        ), "All moments must have a period to figure out their offset"
        assert moment.instant_time_ms is not None, "You promised this would all be set"
        period_offset_ms = period_offsets[int(period_num)]
        yield moment.update(
            start_time_ms=moment.start_time_ms + period_offset_ms,
            end_time_ms=moment.end_time_ms + period_offset_ms,
            instant_time_ms=moment.instant_time_ms + period_offset_ms,
        )


@dataclass
class LineupPlayer(DataClassJsonMixin):
    id: str
    jersey: str
    position: str
    minutes: float
    max_speed: float
    max_acceleration: float
    tracking_idx: int


@dataclass
class ATVRawFeedInfo(DataClassJsonMixin):
    angle: str
    projection_matrix: List[float]
    video_location: Optional[str]


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class EFDGVideoRecord(VideoRecord, DataClassJsonMixin):
    efdg_fixture_id: str
    home_team_id: str
    away_team_id: str
    home_lineup: List[LineupPlayer]
    away_lineup: List[LineupPlayer]
    periods: List[Period]
    player_tracking_manifest: Optional[str] = None
    ball_tracking_manifest: Optional[str] = None
    broadcast_media_stream_id: Optional[str] = None
    broadcast_video_location: Optional[str] = None
    tactical_video_location: Optional[str] = None
    raw_feeds: List[ATVRawFeedInfo] = field(default_factory=list)
    broadcast_video_opposite_from_tracking: bool = False

    _extra_properties: CatchAll = field(default_factory=dict)

    @property
    def adjusted_ground_truth(self) -> List[Moment]:
        """The ground truth moments on these objects all have their times set
        relative to the period they belong to.

        This list will instead be relative to the broadcast video
        """
        if self.ground_truth_moments is None:
            raise ValueError(f"No ground truth moments for {self.video_id}")
        return list(
            adjust_moments_against_periods(self.ground_truth_moments, self.periods)
        )

_QUILT_CACHE_DIRECTORY = str(Path.home() / ".hudl" / "aml" / "data" / "tracked-moments")
PACKAGE_NAME = "hudlrd/efdg_moments_and_tracking"

class EFDGMomentsDataset(MomentDataset[EFDGVideoRecord]):
    def __init__(
        self,
        top_hash: Optional[str] = None,
        writable: bool = False,
        download_missing_package: bool = True,
    ):
        super().__init__(
            PACKAGE_NAME,
            writable,
            top_hash,
            _QUILT_CACHE_DIRECTORY,
            download_missing_package,
        )

def download_from_s3(s3, bucket_name, key_name, outfilepath):
    with open(outfilepath, 'wb') as f:
        s3.download_fileobj(bucket_name, key_name, f)

BASE_FOLDER = Path("datasets/efdg/")

def download_and_convert_to_pickle(efdg_data):
    
    Path.mkdir(BASE_FOLDER, exist_ok=True)

    for video_id in tqdm(efdg_data.video_ids, desc="Converting to chunks"):
        stream_s3_path = efdg_data.get_video(video_id).player_tracking_manifest
        if stream_s3_path is None:
            continue
        VIDEO_FOLDER = BASE_FOLDER / video_id
        if (VIDEO_FOLDER / 'period_1.pkl').is_file():
            continue
        Path.mkdir(VIDEO_FOLDER, exist_ok=True)
        s3_path_split = stream_s3_path.split('/')
        bucket_name = s3_path_split[2]
        manifest_key_name = '/'.join(s3_path_split[3:])
        s3 = boto3.client('s3')

        download_from_s3(s3, bucket_name, manifest_key_name, VIDEO_FOLDER / "manifest.gcu8")

        video_chunks = {}
        portion = 0
        with open(VIDEO_FOLDER / "manifest.gcu8") as f:
            lines = f.readlines()
        for line in lines:
            if not line.startswith("#"):
                chunk_filename = line.rstrip("\n")
                chunk_key_name = "/".join(manifest_key_name.split("/")[:-1] + [chunk_filename])
                download_from_s3(s3, bucket_name, chunk_key_name, VIDEO_FOLDER / chunk_filename)
                with gzip.open(VIDEO_FOLDER / chunk_filename, 'r') as fin:
                    chunk_data = json.loads(fin.read().decode('utf-8'))
                curr_chunk = Chunk.from_dict(chunk_data)
                if len(video_chunks) == 0 or curr_chunk.delta != video_chunks[portion][-1].delta:
                    portion += 1
                    video_chunks[portion] = []
                video_chunks[portion].append(curr_chunk)

        final_chunks = []
        for chunks_list in video_chunks.values():
            final_chunks.append(time_concat(chunks_list))
        
        for period, idx in [(1, 1), (2, 4)]:
            with open(VIDEO_FOLDER / f'period_{period}.pkl', 'wb') as handle:
                pickle.dump(final_chunks[idx], handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_dataset(efdg_data, max_seq_len:int, frame_step:int):
    for video_id in tqdm(efdg_data.video_ids, desc="Convert to Agentformer format"):
        out_folder = BASE_FOLDER / "sequences"
        out_folder.mkdir(exist_ok=True)
        VIDEO_FOLDER = BASE_FOLDER / video_id
        total_seq_len = max_seq_len * frame_step
        for period in [1, 2]:
            if not (VIDEO_FOLDER / f'period_{period}.pkl').is_file():
                continue
            with open(VIDEO_FOLDER / f'period_{period}.pkl', 'rb') as handle:
                chunk = pickle.load(handle)
            for seq_idx in range(0, chunk.tracking.shape[0], total_seq_len):
                all_data = []
                for t_id, t_idx in chunk.trackables.items():
                    new_data = np.ones([chunk.tracking[seq_idx:seq_idx+total_seq_len:frame_step].shape[0], 17]) * -1.0
                    new_data[:, 0] = np.arange(new_data.shape[0])
                    new_data[:, 1] = t_idx
                    new_data[:, [13, 15]] = chunk.tracking[seq_idx:seq_idx+total_seq_len:frame_step, t_idx, :2]
                    # new_data[chunk.tracking[:, t_idx, 2] != 1] = np.nan
                    new_data = new_data[~np.any(np.isnan(new_data), axis=1)]
                    all_data.append(new_data)
                all_data = np.concatenate(all_data, axis=0)
                all_data = all_data.astype(str)
                all_data[:, 2] = 'Pedestrian'
                np.savetxt(out_folder / f"{video_id}_{period}_{seq_idx}.txt", all_data, fmt='%s')

def main(max_seq_len:int=5000, frame_step:int=25):
    efdg_data = EFDGMomentsDataset()
    download_and_convert_to_pickle(efdg_data)
    create_dataset(efdg_data, max_seq_len, frame_step)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire(main)
