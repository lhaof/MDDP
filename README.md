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

## Pre-training

### Data Pre-process
For diffusion based pre-training, training patches are selected using [Yottixel](https://github.com/KimiaLabMayo/yottixel)-related method:
```
python HE_IHC_select.py
```

### Mddp model training
```
python pretraining/main.py --config configs/pretrain.yaml --train --sample_at_start --save_top --gpu_ids 0
```
The pre-trained weights can be downloaded in Baiduyun Link：https://pan.baidu.com/s/18Vxkcu9Jk_6WM2ocjpA2Kg?pwd=kued
## WSI classification
### Feature Extraction
In this step, we extract the Camelyon16 features using the pre-trained mddp model. Firstly, you need to extract patches following the [CLAM](https://github.com/mahmoodlab/CLAM). 
```
python clam_cls/create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 
```
Then, modify the 'csv_path', 'data_h5_dir', 'data_slide_dir', 'feat_dir' to your local directories in 'feature_extraction/runners/DiffusionBasedModelRunners/BBDMRunner.py'. The features are extracted using command:
```
python main.py --config configs/pretrained.yaml --sample_to_eval --gpu_ids 0 --resume_model MDDP_pretrained/pretrained_weights.pth
```

### WSI classification
```
python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code task_1_tumor_vs_normal_CLAM_100 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir CLAM_features/ --data_MDDP_dir mddp_features/ --results_dir logs/
```
### Evaluation
```
python eval.py --drop_out --fold 1 --splits_dir camelyon16/ --models_exp_code task_1_tumor_vs_normal_CLAM_100_s1 --save_exp_code task_1_tumor_vs_normal_CLAM_100_s1_cv --task task_1_tumor_vs_normal --model_type clam_sb --results_dir logs/ --data_root_dir CLAM_features/ --data_BBDM_dir mddp_features/
```

## Citation

If any part of this code is used, please give appropriate citations to our paper. <br />

BibTex entry: <br />
```
@inproceedings{lou2024multi,
  title={Multi-modal Denoising Diffusion Pretraining for Whole-Slide Image Classification},
  author={Lou, Wei and Li, Guanbin and Wan, Xiang and Li, Haofeng},
  booktitle={ACM Multimedia 2024}
}
```
