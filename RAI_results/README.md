# Rethinking Rotation in Self-Supervised Contrastive Learning: Adaptive Positive or Negative Data Augmentation

## Explanation about .pkl files
These .pkl files indicate RAI (rotation positive) or not-RAI (rotation negative) flag.  
These .pkl files consist of the values 0 or 1.  
RAI (rotation positive): 1  
not-RAI (rotation negative): 0

The index of each .pkl corresponds to the index of the sample in each dataset.  

## Creation
Please refer to `RAI_SAMPLING/inference.py`.
## Usage
Please refer to `PNDA_MOCO/main_moco_pnda.py` or `PNDA_SIMCLR/main_simclr_pnda.py`.