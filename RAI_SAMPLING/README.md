# Rethinking Rotation in Self-Supervised Contrastive Learning: Adaptive Positive or Negative Data Augmentation
<p align="center">
    <img src=../figure/rai_sampling.jpg width="900"> 
</p>

## 1. Requirements

### Environments

Currently, requires following packages
- python 3.7
- torch 1.4
- torchvision 0.5
- CUDA 10.1+
- tensorboard 2.0
- tensorboardX
- scipy
- tqdm

We have done these codes for RAI-sampling with a single Nvidia V100 GPU.

## 2. Training

Please refer to `run_train.sh`

If you encounter an assert error ("It is likely that this model are overfitting, please try again."), run the above command again.  
We conducted 3 runs and chose the model that best matched the criterion in Section 3.4, i.e., the model whose accuracy after step 1 and step 2 are close.

## 3. Inference

Please refer to `run_inference.sh`

In this step, RAI-samling result is output as `{P.dataset}_nega_posi_flag_{time.time()}.pkl`. For the explanation of this .pkl file, please refer to `../RAI_results`


## Pretrained models

Our pretrained models are available [here](https://drive.google.com/drive/folders/1GOWjenEhZgYpxXXSFJnbkN6WVeGiIgki?usp=sharing).
