import os
import random
import torch
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor 
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from util.perceptual import PerceptualHashValue
def fid(mn1, cov1, mn2, cov2, eps=1e-6):
    mn1 = np.atleast_1d(mn1)
    mn2 = np.atleast_1d(mn2)
    
    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)
    
    diff = mn1 - mn2
        
    # product might be almost singular
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn(("fid() got singular product; adding {} to diagonal of "
                       "cov estimates").format(eps))
        offset = np.eye(d) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * tr_covmean
targ_dir = '/mntnfs/med_lihaofeng/virtual_stain/CRC_dataset/testB_256'
pred_dir = '/mntnfs/med_lihaofeng/virtual_stain/logs/BBDM/CRC/BrownianBridge/sample_to_eval/results/'

img_list = [f for f in os.listdir(pred_dir) if f.endswith(('png', 'jpg'))]
img_format = '.' + img_list[0].split('.')[-1]
img_list = [f.replace('.png', '').replace('.jpg', '') for f in img_list]
random.seed(0)
random.shuffle(img_list)

# PHV statistics
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
PHV = PerceptualHashValue(
        T=0.01, network='resnet50', layers=layers, 
        resize=False, resize_mode='bilinear',
        instance_normalized=False).to(device)
all_phv = []
for i in tqdm(img_list):
    fake = io.imread(os.path.join(pred_dir, i + img_format))
    real = io.imread(os.path.join(targ_dir, i + img_format))

    fake = to_tensor(fake).to(device)
    real = to_tensor(real).to(device)

    phv_list = PHV(fake, real)
    all_phv.append(phv_list)
all_phv = np.array(all_phv)
all_phv = np.mean(all_phv, axis=0)
res_str = ''
for layer, value in zip(layers, all_phv):
    res_str += f'{layer}: {value:.4f} '
print(res_str)
print(np.round(all_phv, 4))
'''
# FID statistics
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
num_avail_cpus = len(os.sched_getaffinity(0))
num_workers = min(num_avail_cpus, 8)

real_paths = [os.path.join(targ_dir, f + img_format) for f in img_list]
fake_paths = [os.path.join(pred_dir, f + img_format) for f in img_list]
print(f"Total number of images: {len(real_paths)}")

dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)

m1, s1 = calculate_activation_statistics(real_paths, model, batch_size=10, dims=dims,
                                    device=device, num_workers=num_workers)

m2, s2 = calculate_activation_statistics(fake_paths, model, batch_size=10, dims=dims,
                                    device=device, num_workers=num_workers)

fid_value = calculate_frechet_distance(m1, s1, m2, s2)
fid_value = fid(m1, s1, m2, s2)
print(f'FID: {fid_value:.2f}')

# KID statistics
command = f'python util/kid_score.py --true {targ_dir} --fake {pred_dir}'
os.system(command)
'''
# PSNR and SSIM statistics
psnr = []
ssim = []
for i in tqdm(img_list):
    fake = io.imread(os.path.join(pred_dir, i + img_format))
    real = io.imread(os.path.join(targ_dir, i + img_format))
    PSNR = peak_signal_noise_ratio(fake, real)
    psnr.append(PSNR)
    print(fake.shape, real.shape)
    SSIM = structural_similarity(fake, real, multichannel=True,channel_axis=-1)
    ssim.append(SSIM)
average_psnr = sum(psnr)/len(psnr)
average_ssim = sum(ssim)/len(ssim)
print(pred_dir)
print("The average psnr is " + str(average_psnr))
print("The average ssim is " + str(average_ssim))
print(f"{average_psnr:.4f} {average_ssim:.4f}")
'''