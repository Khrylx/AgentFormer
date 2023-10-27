from pathlib import Path


def get_efdg_split(data_root: str):
    seq_names = [x.stem for x in Path(data_root).glob('**/*') if x.is_file()]
    seq_videos = {}
    for x in seq_names:
        video_id = x.split("_")[0]
        if video_id not in seq_videos:
            seq_videos[video_id] = []
        seq_videos[video_id].append(x)
    
    train, val, test = [], [], []
    for idx, video_id in enumerate(seq_videos.keys()):
        if idx % 10 < 6:
            train += seq_videos[video_id]
        elif idx % 10 < 8:
            val += seq_videos[video_id]
        else:
            test += seq_videos[video_id]
    print(len(train), len(val), len(test))
    return train, val, test
