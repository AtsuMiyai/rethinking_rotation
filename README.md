# Rethinking Rotation in Self-Supervised Contrastive Learning: Adaptive Positive or Negative Data Augmentation


This is the official PyTorch impelementation of our paper "Rethinking Rotation in Self-Supervised Contrastive Learning: Adaptive Positive or Negative Data Augmentation" [Miyai+, WACV2023].  

We propose a novel augmentation strategy for self-supervised contrastive learning, **adaptive Positive or Negative Data Augmentation (PNDA)**, 
in which an original and its rotated image are a positive pair if they are semantically close and a negative pair if they are semantically different. To achieve PNDA, we first determine whether rotation is positive or negative on an image-by-image basis in an unsupervised way. Then, we apply PNDA to contrastive learning frameworks. Our experiments showed that PNDA improves the performance of contrastive learning. 

<p align="center">
    <img src=figure/overview_pnda.png width="900"> 
</p>

## Preparation
Please make `data` folder. 
Please download following datasets to `data`.
* [CIFAR100](https://drive.google.com/file/d/1FRi1K1ZQ-OCgIhMROwjYAt_m_K44Q9ea/view?usp=sharing)
* [Tiny ImageNet](https://drive.google.com/file/d/1b846FVuOPpZbOnKaiFd2OL4MZntbiiPr/view?usp=sharing)

## Usage
Our method comprises two major steps:

1. Run sampling of RAI whose rotation is treated as positive.
2. Run PNDA for contrastive learning (MoCo v2, SimCLR).

The codes and their details for each step are stored in a sub-directory with the corresponding name `RAI_SAMPLING`, `PNDA_SIMCLR`, `PNDA_MOCO`.

The RAI sampling results used in the experiment are stored in `RAI_results`.
Therefore, you can start from either step.


## Citation
@inproceedings{miyai2023pnda,  
  title={Rethinking Rotation in Self-Supervised Contrastive Learning: Adaptive Positive or Negative Data Augmentation},  
  author={Miyai, Atsuyuki and Yu, Qing and Ikami, Daiki and Irie, Go and Aizawa Kiyoharu},  
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},  
  year={2023}  
}