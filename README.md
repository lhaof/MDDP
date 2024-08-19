# Multi-modal Denoising Diffusion Pre-training for Whole-Slide Image Classification
This is the official PyTorch implementation of MDDP, a diffusion-based pretraining method for WSI classification.
![](pictures/mddp.png)

## Set Up Environment
```
conda env create -f environment.yml
conda activate mddp
pip install torch==1.13.1 torchvision==0.14.1
```

# Running the Code

## pre-training

### Data Format
For diffusion based pre-training, training patches are selected using [Yottixel](https://github.com/KimiaLabMayo/yottixel)-related method:
```
python HE_IHC_select.py --gpu='0' 
```
