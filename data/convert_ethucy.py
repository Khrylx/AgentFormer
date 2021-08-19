import os
import numpy as np
import glob
import sys
import subprocess
import argparse
sys.path.append(os.getcwd())


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="eth")
parser.add_argument('--raw_path', default="eth_data")
parser.add_argument('--out_path', default="datasets/eth_ucy")
args = parser.parse_args()

for mode in ['train', 'test', 'val']:
    raw_files = sorted(glob.glob(f'{args.raw_path}/{args.dataset}/{mode}/*.txt'))
    for raw_file in raw_files:
        raw_data = np.loadtxt(raw_file)
        new_data = np.ones([raw_data.shape[0], 17]) * -1.0
        new_data[:, 0] = raw_data[:, 0] / 10
        new_data[:, 1] = raw_data[:, 1]
        new_data[:, [13, 15]] = raw_data[:, 2:4]
        new_data = new_data.astype(np.str)
        new_data[:, 2] = 'Pedestrian'
        out_file = f'{args.out_path}/{args.dataset}/{os.path.basename(raw_file)}'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        np.savetxt(out_file, new_data, fmt='%s')
        print(out_file)