from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
from zmq import device

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

	img_orig = img_lab_orig[:,:,0]
	img_rs_l = img_lab_rs[:,:,0:]
	img_rs_ab = img_lab_rs[:,:,1:]

	tens_orig = torch.Tensor(img_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_rs_l)[None,None,:,:]
	tens_rs_ab = torch.Tensor(img_rs_ab)[None,None,:,:]

	return (tens_orig, tens_rs_l, tens_rs_ab)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

def rgb2yuv_601(rgb):
	mat = torch.tensor([[0.299, -0.14714119, 0.61497538],
						[0.587, -0.28886916, -0.51496512],
						[0.114, 0.43601035, -0.10001026]], device=rgb.device)
	rgb_ = rgb.transpose(1, 3)
	yuv = torch.tensordot(rgb_, mat, dims=1).transpose(1, 3)
	return yuv

def yuv2rgb_601(y, uv):
	mat = torch.tensor([[1, 1, 1],
						[0, -0.39465, 2.03211],
						[1.13983, -0.58060, 0]], device=y.device)
	yuv = torch.cat([y, uv], dim=1).transpose(1, 3)
	rgb = torch.tensordot(yuv, mat, dims=1).transpose(1, 3)
	return rgb

if __name__ == '__main__':
	mat = torch.rand(2,3,2,2)
	yuv = rgb2yuv_601(mat)[:, 0:1].repeat(1, 3, 1, 1)
	print(mat, yuv)