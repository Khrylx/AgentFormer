import torch
import os
import numpy as np
import copy
import cv2
import glob

class TepperPreprocessor(object):
    def __init__(self, data_root, seq_name, parser, log, split='train', phase='training'):
        tepper_path = "./datasets/tepper/Pedestrian_labels/0_frame.txt"
        raw_data = np.genfromtxt(tepper_path, delimiter=',')
        self.seq_name = '0'

        self.gt = raw_data
        frames = self.gt[:, 0].astype(np.float32).astype(int)
        fr_start, fr_end = frames.min(), frames.max()
        self.init_frame = fr_start
        self.num_fr = fr_end + 1 - fr_start
        self.past_frames = 8
        self.fut_frames = 12
        self.frame_skip = 1
        self.min_past_frames = 8
        self.min_future_frames = 12 

    def get_id(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())
        return id

    def get_valid_id(self, pre_data, fut_data):
        cur_id = self.get_id(pre_data[0])
        valid_id = []
        for idx in cur_id:
            exist_pre = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in pre_data[:min_past_frames]]
            exist_fut = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in fut_data[:min_future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                valid_id.append(idx)
        return valid_id
    
    def get_history_data(self, frame_idx):
        data_list = []
        for i in range(self.past_frames):
            if frame_idx - i < self.init_frame:
                data = []
            data = self.gt[self.gt[:, 0] == (frame_idx - i) * self.frame_skip]
            data_list.append(data)
        return data_list
    
    def get_future_data(self, frame_idx):
        data_list = []
        for i in range(1, self.fut_frames + 1):
            data = self.gt[self.gt[:, 0] == frame_idx  + i * self.frame_skip]
            data_list.append(data)
        return data_list

    def get_history_motion(self, data_list, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.past_frames)
            past_coords = torch.zeros(self.past_frames, 3)
            for frame_idx in range(self.past_frames):
                past_data = data_list[frame_idx]
                if len(past_data) > 0 and identity in past_data[:, 1]:
                    # Keep all indices (x, y, z), don't apply traj scale.
                    found_data = past_data[past_data[:, 1] == identity].squeeze()
                    past_coords[self.past_frames-1-frame_idx, :] = torch.from_numpy(found_data[[2,3,4]]).float()
                    mask_i[self.past_frames - 1 - frame_idx] = 1.0
                elif frame_idx > 0:
                    past_coords[self.past_frames-1 - frame_idx, :] = past_coords[self.past_frames - frame_idx, :]
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(past_coords)
            mask.append(mask_i)
        
        return motion, mask

    def get_future_motion(self, data_list, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.fut_frames)
            pos_3d = torch.zeros([self.fut_frames, 3])
            for j in range(self.fut_frames):
                fut_data = data_list[j]              # cur_data
                if len(fut_data) > 0 and identity in fut_data[:, 1]:
                    found_data = fut_data[fut_data[:, 1] == identity].squeeze()[[2,3,4]]
                    pos_3d[j, :] = torch.from_numpy(found_data).float()
                    mask_i[j] = 1.0
                elif j > 0:
                    pos_3d[j, :] = pos_3d[j - 1, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(pos_3d)
            mask.append(mask_i)
        
        return motion, mask
    
    def __call__(self, frame_idx):
        assert frame_idx - self.init_frame >= 0 and frame_idx - self.init_frame < self.num_fr

        pre_data = self.get_history_data(frame_idx)
        fut_data = self.get_future_data(frame_idx)
        valid_id = self.get_valid_id(pre_data, fut_data)

        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            return None
        pred_mask = None
        heading = None
        
        pre_motion_3D, pre_motion_mask = self.get_history_motion(pre_data, valid_id)
        fut_motion_3D, fut_motion_mask = self.get_future_motion(fut_data, valid_id)

        data = {
            'pre_motion_3D': pre_motion_3D,
            'fut_motion_3D': fut_motion_3D,
            'fut_motion_mask': fut_motion_mask,
            'pre_motion_mask': pre_motion_mask,
            'pre_data': pre_data,
            'fut_data': fut_data,
            'heading': heading,
            'valid_id': valid_id,
            'pred_mask': pred_mask,
            'scene_map': None,
            'seq': self.seq_name,
            'frame': frame_idx
        }

        return data