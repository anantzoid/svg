import os
import io
from scipy.misc import imresize 
import numpy as np
from PIL import Image
from scipy.misc import imresize, imread, imsave
import tarfile

folder = '/misc/vlgscratch4/FergusGroup/anant/EPIC_KITCHENS_2018/'
tar_location = 'frames_rgb_flow/rgb/train/'
dest_folder = os.path.join(folder, 'train')
global_path = os.path.join(folder, tar_location)

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

all_local_dest = []
for p_folder in os.listdir(global_path):
    #P01/..
    for p_tar in os.listdir(os.path.join(global_path, p_folder)):
        #P01/P1_01.tar
        print(os.path.join(global_path, p_folder, p_tar))
        local_dest_folder = os.path.join(dest_folder, p_folder, p_tar.split(".")[0])
        all_local_dest.append(local_dest_folder)
        if not os.path.exists(local_dest_folder):
            os.makedirs(local_dest_folder)

            tar = tarfile.open(os.path.join(global_path, p_folder, p_tar))
            tar.extractall(path=local_dest_folder)
            tar.close()

for img_folder in all_local_dest:
    print(img_folder)
    processed_dest = img_folder.replace("train", "train_processed")

    if not os.path.exists(processed_dest):
        os.makedirs(processed_dest)
    print(processed_dest)
    for imgs in os.listdir(img_folder):
        im = imread(os.path.join(img_folder, imgs))
        im = imresize(im[:, 100:356, :], (128, 128, 3))
        imsave(os.path.join(processed_dest, imgs), im)
