# AgentFormer
This repo contains the official implementation of our paper:
  
AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting  
Ye Yuan, Xinshuo Weng, Yanglan Ou, Kris Kitani  
**ICCV 2021**  
[[website](https://www.ye-yuan.com/agentformer)] [[paper](https://arxiv.org/abs/2103.14023)]

# Overview
![Loading AgentFormer Overview](https://github.com/Khrylx/khrylx.github.io/blob/main/agentformer/data/overview.png "AgentFormer Overview")

# Important Note
We have recently noticed a [normalization bug](https://github.com/Khrylx/AgentFormer/issues/5) in the code and after fixing it, the performance of our method is worse than the original numbers reported in the ICCV paper. For comparision, please use the correct numbers in the updated [arXiv version](https://arxiv.org/abs/2103.14023).

# Installation 

### Environment
* **Tested OS:** MacOS, Linux
* Python >= 3.7
* PyTorch == 1.8.0
### Dependencies:
1. Install [PyTorch 1.8.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

### Datasets
* For the ETH/UCY dataset, we already included a converted version compatible with our dataloader under [datasets/eth_ucy](datasets/eth_ucy).
* For the nuScenes dataset, the following steps are required:
  1. Download the orignal [nuScenes](https://www.nuscenes.org/nuscenes) dataset. Checkout the instructions [here](https://github.com/nutonomy/nuscenes-devkit).
  2. Follow the [instructions](https://github.com/nutonomy/nuscenes-devkit#prediction-challenge) of nuScenes prediction challenge. Download and install the [map expansion](https://github.com/nutonomy/nuscenes-devkit#map-expansion).
  3. Run our [script](data/process_nuscenes.py) to obtain a processed version of the nuScenes dataset under [datasets/nuscenes_pred](datasets/nuscenes_pred):
      ```
      python data/process_nuscenes.py --data_root <PATH_TO_NUSCENES>
      ``` 
### Pretrained Models
* You can download pretrained models from [Google Drive](https://drive.google.com/file/d/1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1b6Ms_aY4U0fhCe5C8cznVQ) (password: 9rvb) to reproduce the numbers in the paper.
* Once the `agentformer_models.zip` file is downloaded, place it under the root folder of this repo and unzip it:
  ```
  unzip agentformer_models.zip
  ```
  This will place the models under the `results` folder. Note that the pretrained models directly correspond to the config files in [cfg](cfg).


# Evaluation
### ETH/UCY
Run the following command to test pretrained models for the ETH dataset:
```
python test.py --cfg eth_agentformer --gpu 0
```
You can replace `eth` with {`hotel`, `univ`, `zara1`, `zara2`} to test other datasets in ETH/UCY. You should be able to get the numbers reported in the paper as shown in this table:
| Ours  | ADE  | FDE  |
|-------|------|------|
| ETH   | 0.45 | 0.75 |
| Hotel | 0.14 | 0.22 |
| Univ  | 0.25 | 0.45 |
| Zara1 | 0.18 | 0.30 |
| Zara2 | 0.14 | 0.24 |
|  Avg  | 0.23 | 0.39 |

### nuScenes
Run the following command to test pretrained models for the nuScenes dataset:
```
python test.py --cfg nuscenes_5sample_agentformer --gpu 0
```
You can replace `5sample` with `10sample` to compute all the metrics (ADE_5, FDE_5, ADE_10, FDE_10). You should be able to get the numbers reported in the paper as shown in this table:
|       | ADE_5 | FDE_5 | ADE_10 | FDE_10 |
|-------|-------|-------|--------|--------|
| Ours  | 1.856 | 3.889 |  1.452 |  2.856 |

# Training
You can train your own models with your customized configs. Here we take the ETH dataset as an example, but you can train models for other datasets with their corresponding [configs](cfg). AgentFormer requires **two-stage** training:
1. Train the AgentFormer VAE model (everything but the trajectory sampler):
    ```
    python train.py --cfg user_eth_agentformer_pre --gpu 0
    ```
2. Once the VAE model is trained, train the AgentFormer DLow model (trajectory sampler):
    ```
    python train.py --cfg user_eth_agentformer --gpu 0
    ```
    Note that you need to change the `pred_cfg` field in `user_eth_agentformer` to the config you used in step 1 (`user_eth_agentformer_pre`) and change the `pred_epoch` to the VAE model epoch you want to use.


# Citation
If you find our work useful in your research, please cite our paper [AgentFormer](https://www.ye-yuan.com/agentformer/):
```bibtex
@inproceedings{yuan2021agent,
    title={AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting},
    author={Yuan, Ye and Weng, Xinshuo and Ou, Yanglan and Kitani, Kris},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2021}
}
```

# License
Please see the [license](LICENSE) for further details.
