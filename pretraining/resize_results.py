 # -*- coding: utf-8 -*-
import Augmentor
import glob
import numpy as np
import os
import random
import PIL
import cv2
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
 
folder_path = '/mntnfs/med_lihaofeng/virtual_stain/CRC_dataset/testB/'

train_save_path = '/mntnfs/med_lihaofeng/virtual_stain/CRC_dataset/testB_256/'


files = os.listdir(folder_path)
for file in files:
    image = cv2.imread(folder_path + file).astype(np.uint8)
    resize_shape = (256,256)

    print(resize_shape)
    resize_image = cv2.resize(image,resize_shape,interpolation=cv2.INTER_CUBIC).astype(np.uint8)

    cv2.imwrite(train_save_path +file,resize_image)
'''
folders = os.listdir(folder_path)

for folder in folders:
    print(folder+ '.png')
    image = cv2.imread(folder_path + folder + '/output_0.png').astype(np.uint8)
    h,w = image.shape[0],image.shape[1]
    print(h,w)

    resize_shape = (1024,1024)

    print(resize_shape)
    resize_image = cv2.resize(image,resize_shape,interpolation=cv2.INTER_CUBIC).astype(np.uint8)

    cv2.imwrite(train_save_path + folder + '.png',resize_image)

'''
