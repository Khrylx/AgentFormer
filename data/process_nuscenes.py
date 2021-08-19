import json
import os
from pyquaternion import Quaternion
from itertools import chain
from typing import List
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
import cv2
import argparse


NUM_IN_TRAIN_VAL = 200
past_frames = 4
future_frames = 12


def get_prediction_challenge_split(split: str, dataroot: str) -> List[str]:
    """
    Gets a list of {instance_token}_{sample_token} strings for each split.
    :param split: One of 'mini_train', 'mini_val', 'train', 'val'.
    :param dataroot: Path to the nuScenes dataset.
    :return: List of tokens belonging to the split. Format {instance_token}_{sample_token}.
    """
    if split not in {'mini_train', 'mini_val', 'train', 'train_val', 'val'}:
        raise ValueError("split must be one of (mini_train, mini_val, train, train_val, val)")
    
    if split == 'train_val':
        split_name = 'train'
    else:
        split_name = split

    path_to_file = os.path.join(dataroot, "maps", "prediction", "prediction_scenes.json")
    prediction_scenes = json.load(open(path_to_file, "r"))
    scenes = create_splits_scenes()
    scenes_for_split = scenes[split_name]
    
    if split == 'train':
        scenes_for_split = scenes_for_split[NUM_IN_TRAIN_VAL:]
    if split == 'train_val':
        scenes_for_split = scenes_for_split[:NUM_IN_TRAIN_VAL]

    token_list_for_scenes = map(lambda scene: prediction_scenes.get(scene, []), scenes_for_split)

    return prediction_scenes, scenes_for_split, list(chain.from_iterable(token_list_for_scenes))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help="path to the original nuScenes dataset")
    parser.add_argument('--data_out', default='datasets/nuscenes_pred/', help="path where you save the processed data")
    args = parser.parse_args()

    DATAROOT = args.data_root
    DATAOUT = args.data_out

    nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)
    map_version = '0.1'
    splits = ['train', 'val', 'test']

    for split in splits:
        os.makedirs(f'{DATAOUT}/label/{split}', exist_ok=True)
        os.makedirs(f'{DATAOUT}/map_{map_version}', exist_ok=True)
        if split == 'val':
            key = 'train_val'
        elif split == 'test':
            key = 'val'
        else:
            key = 'train'
        prediction_scenes, split_scenes, split_data = get_prediction_challenge_split(key, dataroot=DATAROOT)
        helper = PredictHelper(nuscenes)

        total_pred = 0

        for scene_name in split_scenes:
            scene_token = nuscenes.field2token('scene', 'name', scene_name)[0]
            scene = nuscenes.get('scene', scene_token)
            scene_data_orig = prediction_scenes.get(scene_name, [])
            if len(scene_data_orig) == 0:
                continue
            scene_data_orig_set = set(scene_data_orig)
            scene_data = set(scene_data_orig)
            for data in scene_data_orig:
                cur_sample = helper.get_sample_annotation(*data.split('_'))
                sample = cur_sample
                for i in range(past_frames - 1):
                    if sample['prev'] == '':
                        break
                    sample = nuscenes.get('sample_annotation', sample['prev'])
                    cur_data = sample['instance_token'] + '_' + sample['sample_token']
                    scene_data.add(cur_data)
                sample = cur_sample
                for i in range(future_frames):
                    sample = nuscenes.get('sample_annotation', sample['next'])
                    cur_data = sample['instance_token'] + '_' + sample['sample_token']
                    scene_data.add(cur_data)

            all_tokens = np.array([x.split("_") for x in scene_data])
            all_samples = set(np.unique(all_tokens[:, 1]).tolist())
            all_instances = np.unique(all_tokens[:, 0]).tolist()
            first_sample_token = scene['first_sample_token']
            first_sample = nuscenes.get('sample', first_sample_token)
            while first_sample['token'] not in all_samples:
                first_sample = nuscenes.get('sample', first_sample['next'])

            frame_id = 0
            sample = first_sample
            cvt_data = []
            while True:
                if sample['token'] in all_samples:
                    instances_in_frame = []
                    for ann_token in sample['anns']:
                        annotation = nuscenes.get('sample_annotation', ann_token)
                        category = annotation['category_name']
                        instance = annotation['instance_token']
                        cur_data = instance + '_' + annotation['sample_token']
                        if cur_data not in scene_data:
                            continue
                        instances_in_frame.append(instance)
                        # get data
                        data = np.ones(18) * -1.0
                        data[0] = frame_id
                        data[1] = all_instances.index(instance)
                        data[10] = annotation['size'][0]
                        data[11] = annotation['size'][2]
                        data[12] = annotation['size'][1]
                        data[13] = annotation['translation'][0]
                        data[14] = annotation['translation'][2]
                        data[15] = annotation['translation'][1]
                        data[16] = Quaternion(annotation['rotation']).yaw_pitch_roll[0]
                        data[17] = 1 if cur_data in scene_data_orig_set else 0
                        data = data.astype(str)
                        if 'car' in category:
                            data[2] = 'Car'
                        elif 'bus' in category:
                            data[2] = 'Bus'
                        elif 'truck' in category:
                            data[2] = 'Truck'
                        elif 'emergency' in category:
                            data[2] = 'Emergency'
                        elif 'construction' in category:
                            data[2] = 'Construction'
                        else:
                            raise ValueError(f'wrong category {category}')
                        cvt_data.append(data)

                frame_id += 1
                if sample['next'] != '':
                    sample = nuscenes.get('sample', sample['next'])
                else:
                    break
                
            cvt_data = np.stack(cvt_data)

            # Generate Maps
            map_name = nuscenes.get('log', scene['log_token'])['location']
            nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=map_name)
            scale = 3.0
            margin = 75
            xy = cvt_data[:, [13, 15]].astype(np.float32)
            x_min = np.round(xy[:, 0].min() - margin)
            x_max = np.round(xy[:, 0].max() + margin)
            y_min = np.round(xy[:, 1].min() - margin)
            y_max = np.round(xy[:, 1].max() + margin)
            x_size = x_max - x_min
            y_size = y_max - y_min
            patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
            patch_angle = 0
            canvas_size = (np.round(scale * y_size).astype(int), np.round(scale * x_size).astype(int))
            homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
            layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line', 'ped_crossing', 'walkway']
            colors = {
                'rest': [255, 240, 243],
                'lane': [206, 229, 223],
                'road_segment': [206, 229, 223],
                'drivable_area': [206, 229, 223],
                'ped_crossing': [226, 228, 234],
                'walkway': [169, 209, 232],
                'road_divider': [255, 251, 242],
                'lane_divider': [100, 100, 100],
                'stop_line': [0, 255, 255],
            }

            map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(np.uint8)
            map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
            map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)

            # map for visualization
            map_mask_plot = np.ones_like(map_mask[:3])
            map_mask_plot[:] = np.array(colors['rest'])[:, None, None]
            for layer in ['lane', 'road_segment', 'drivable_area', 'road_divider', 'ped_crossing', 'walkway']:
                xind, yind = np.where(map_mask[layer_names.index(layer)])
                map_mask_plot[:, xind, yind] = np.array(colors[layer])[:, None]

            meta = np.array([x_min, y_min, scale])
            np.savetxt(f'{DATAOUT}/map_{map_version}/meta_{scene_name}.txt', meta, fmt='%.2f')
            cv2.imwrite(f'{DATAOUT}/map_{map_version}/{scene_name}.png', np.transpose(map_mask_vehicle, (1, 2, 0)))
            cv2.imwrite(f'{DATAOUT}/map_{map_version}/vis_{scene_name}.png', cv2.cvtColor(np.transpose(map_mask_plot, (1, 2, 0)), cv2.COLOR_RGB2BGR))

            pred_num = int(cvt_data[:, -1].astype(np.float32).sum())
            assert pred_num == len(scene_data_orig)
            total_pred += pred_num

            np.savetxt(f'{DATAOUT}/label/{split}/{scene_name}.txt', cvt_data, fmt='%s')
            print(f'{scene_name} finished! map_shape {map_mask_plot.shape}')
        
        print(f'{split}_len: {len(split_data)} total_pred: {total_pred}')
