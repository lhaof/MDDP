#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import openslide, numpy as np
import cv2
from cv2 import filter2D
import os
import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/ops.py
def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,3)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,:,0] / (D) - 1.0
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,3)
    cx = np.expand_dims(cx,3)
    cy = np.expand_dims(cy,3)
            
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD


def clean_thumbnail(thumbnail):
    thumbnail_arr = np.asarray(thumbnail)
    
    # writable thumbnail
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]

    # Remove pen marking here
    # We are skipping this
    
    # This  section sets regoins with white spectrum as the backgroud regoin
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    return wthumbnail


HE_path = 'HE/'
IHC_path = '/IHC/'

save_HE_path = 'HE_cropped/'
save_IHC_path = 'IHC_cropped/'


files = os.listdir(IHC_path)

for file in files:
    he_file = file[:8] + '-HE.svs'
    slide = openslide.open_slide(HE_path+ he_file)
    ihc_slide = openslide.open_slide(IHC_path+ file)
    if 'openslide.objective-power' in slide.properties.keys():
        objective_power = int(slide.properties['openslide.objective-power'])
        print(objective_power)
        #patch_size = (objective_power/20.)*1000
        [m, n] = slide.dimensions
        #wsi = np.array(slide.read_region((0, 0), 0, (m, n)))
        #slide.get_thumbnail((300, 300))
        thumbnail = slide.get_thumbnail((500, 500))
        cthumbnail = clean_thumbnail(thumbnail)
        tissue_mask = (cthumbnail.mean(axis=2) != 255)*1.

        objective_power = int(slide.properties['openslide.objective-power'])
        w, h = slide.dimensions

        # at 20x its 1000x1000
        patch_size = (objective_power/20.)*1024

        mask_hratio = (tissue_mask.shape[0]/h)*patch_size
        mask_wratio = (tissue_mask.shape[1]/w)*patch_size


        # iterating over patches
        patches = []

        for i, hi in enumerate(range(0, h, int(patch_size) )):

            _patches = []
            for j, wi in enumerate(range(0, w, int(patch_size) )):

                # check if patch contains 70% tissue area
                mi = int(i*mask_hratio)
                mj = int(j*mask_wratio)

                patch_mask = tissue_mask[mi:mi+int(mask_hratio), mj:mj+int(mask_wratio)]

                tissue_coverage = np.count_nonzero(patch_mask)/patch_mask.size

                _patches.append({'loc': [i, j], 'wsi_loc': [int(hi), int(wi)], 'tissue_coverage': tissue_coverage})

            patches.append(_patches)




        # for patch to be considered it should have this much tissue area
        tissue_threshold = 0.7

        flat_patches = np.ravel(patches)
        for patch in tqdm.tqdm(flat_patches):

            # ignore patches with less tissue coverage
            if patch['tissue_coverage'] < tissue_threshold:
                continue

            # this loc is at the objective power
            h, w = patch['wsi_loc']

            # we will go obe level lower, i.e. (objective power / 4)
            # we still need patches at 5x of size 250x250
            # this logic can be modified and may not work properly for images of lower objective power < 20 or greater than 40
            patch_size_5x = int(((objective_power / 4)/5)*250.)

            patch_region = slide.read_region((w, h), 1, (patch_size_5x, patch_size_5x)).convert('RGB')

            if patch_region.size[0] != 250:
                patch_region = patch_region.resize((250, 250))

            histogram = (np.array(patch_region)/255.).reshape((250*250, 3)).mean(axis=0)
            patch['rgb_histogram'] = histogram    



        selected_patches_flags = [patch['tissue_coverage'] >= tissue_threshold for patch in flat_patches]
        selected_patches = flat_patches[selected_patches_flags]
        if len(selected_patches) > 9:
            kmeans_clusters = 9
            kmeans = KMeans(n_clusters = kmeans_clusters)
            features = np.array([entry['rgb_histogram'] for entry in selected_patches])

            kmeans.fit(features)


            cmap = plt.cm.get_cmap('hsv', kmeans_clusters)
            patch_clusters = np.zeros(np.array(patches).shape+(3,))


            for patch, label in zip(selected_patches, kmeans.labels_):
                patch_clusters[patch['loc'][0], patch['loc'][1], :] = cmap(label)[:3]
                patch['cluster_lbl'] = label


            # Another hyperparameter of Yottixel
            # Yottixel has been tested with 5, 10, and 15 with 15 performing most optimally
            percentage_selected = 25

            mosaic = []

            for i in range(kmeans_clusters):
                cluster_patches = selected_patches[kmeans.labels_ == i]
                n_selected = max(1, int(len(cluster_patches)*percentage_selected/100.))

                km = KMeans(n_clusters=n_selected)
                loc_features = [patch['wsi_loc'] for patch in cluster_patches]
                ds = km.fit_transform(loc_features)

                c_selected_idx = []
                for idx in range(n_selected):
                    sorted_idx = np.argsort(ds[:, idx])

                    for sidx in sorted_idx:
                        if sidx not in c_selected_idx:
                            c_selected_idx.append(sidx)
                            mosaic.append(cluster_patches[sidx])
                            break

            patch_clusters = np.zeros(np.array(patches).shape+(3,))

            for patch in selected_patches:
                patch_clusters[patch['loc'][0], patch['loc'][1], :] = np.array(cmap(patch['cluster_lbl'])[:3])*0.6
            for patch in mosaic:
                patch_clusters[patch['loc'][0], patch['loc'][1], :] = cmap(patch['cluster_lbl'])[:3]

            i = 0
            for patch in mosaic:
                filename = file[:8] + '_' +str(i) + '.png'
            #print(patch['loc'][0]*2000,(patch['loc'][0]+1)*2000,patch['loc'][1]*2000,(patch['loc'][1]+1)*2000)
                he_image = np.array(slide.read_region(((patch['loc'][1])*int(patch_size), patch['loc'][0]*int(patch_size)), 0, (int(patch_size),     int(patch_size))))
                ihc_image = np.array(ihc_slide.read_region(((patch['loc'][1])*int(patch_size), patch['loc'][0]*int(patch_size)), 0, (int(patch_size),     int(patch_size))))
                #print(image.shape)
                he_image = he_image[:,:,:3]
                ihc_image = ihc_image[:,:,:3]
            #image = np.array(wsi)[:,:,:3][:(patch['loc'][0]+1)*int(patch_size),patch['loc'][1]*int(patch_size):(patch['loc'][1]+1)*int(patch_size)]
                he_image = cv2.cvtColor(he_image, cv2.COLOR_RGB2BGR)
                he_image = cv2.resize(he_image,(1024,1024),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(save_HE_path+filename,he_image)
                
                ihc_image = cv2.cvtColor(ihc_image, cv2.COLOR_RGB2BGR)
                ihc_image = cv2.resize(ihc_image,(1024,1024),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(save_IHC_path+filename,ihc_image)
                i += 1
            print('num: ',i)





