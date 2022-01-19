import os
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d','--train_directory', type=str, default='./train_data')
opt = parser.parse_args()

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np

def resize_img(img, HW=(256,256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:,:,0]
    img_l_rs = img_lab_rs[:,:,0]
    img_a_rs = img_lab_rs[:,:,1]
    img_b_rs = img_lab_rs[:,:,2]

    tens_orig_l = img_l_orig
    tens_rs_l = img_l_rs

    return (tens_orig_l, tens_rs_l,img_a_rs, img_b_rs)
names = []
pics = []
labels_a = []
for file in os.listdir(opt.train_directory):
    names.append(file)
print(len(names))
cnt = 0
for name in names:
    if cnt % 1000 == 0:
        print("one")
    cnt += 1
    img = load_img(opt.train_directory+'/'+name)
    (tens_l_orig, tens_l_rs,img_a_rs,img_b_rs) = preprocess_img(img, HW=(256,256))
    pics.append([tens_l_rs])
    labels_a.append([img_a_rs,img_b_rs])

np.save('pics.npy',pics)
np.save('labels.npy',labels_a)