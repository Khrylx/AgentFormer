import logging
import pickle
import numpy as np
import argparse
import os
import sys
import subprocess
import shutil

from torch import Tensor

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing


def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

def save_prediction(pred, data, suffix, save_dir):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = pred[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num

def save_data_for_calibration(data, gt, save_dir, test_name="test_AgentFormer"):
    """
    Pickle provided data for future calibration compute
	Args:
        - test_name
        - tpred_samples : data
        - target_test : gt
    """
    data_for_calibration = {
        "TPRED_SAMPLES": data,
        "TARGET_TEST": gt
    }
    pickle_out_name = os.path.join(save_dir, test_name+".pickle")
    pickle_out = open(pickle_out_name, "wb")
    pickle.dump(data_for_calibration, pickle_out, protocol=2)
    pickle_out.close()
    logging.info("Pickling data for calibration compute...")


def test_model(generator, save_dir, cfg):
    it = 0  # var to help us to count how many times the below generator actually generates something
    samples_motion_3D = []
    gts_motion_3D = []
    while not generator.is_epoch_end():
        data = generator()  # Generator sometimes returns none, is it random?
        if data is None:
            continue
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)  # not all frames arent secuencial
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        sys.stdout.flush()

        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

        samples_motion_3D.append(sample_motion_3D)
        gts_motion_3D.append(gt_motion_3D)

        it = it + 1

    samples_motion_3D = np.array(samples_motion_3D, dtype=Tensor)   # Getting an error here u.u
    gts_motion_3D = np.array(gts_motion_3D, dtype=Tensor)           # Getting an error here u.u
    print(samples_motion_3D.shape)
    print(gts_motion_3D.shape)

    """save samples"""
    save_data_for_calibration(data=sample_motion_3D, gt=gt_motion_3D,save_dir=save_dir)
    print('Generator different from none: %s\r' % (it)) # turns out it=253 at the end

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    # the below epoch gets cut from the model name, in this case if using eth_agentformer_pre.yaml epoch will be only [30]
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = [args.data_eval]

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            if not args.cached:
                test_model(generator, save_dir, cfg)

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)


