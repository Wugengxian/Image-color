from torch.utils.data import Dataset
from torchvision import transforms
from utils.processdata import rgb2yuv_601
from PIL import Image
import random
import os

class yuvdatasets(Dataset):
    def __init__(self,cont_img_path,style_img_path,img_size):
        self.cont_img_path = cont_img_path
        self.style_img_path = style_img_path
        self.img_size = img_size
        self.cont_img_files = self.list_files(self.cont_img_path)
        self.style_img_files = self.list_files(self.style_img_path)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size), Image.BICUBIC),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.cont_img_files)

    def __getitem__(self,idx):
        rgb = Image.open(self.cont_img_files[idx]).convert('RGB')
        rgb = self.transform(rgb)

        return rgb
    
    def getStyle(self):
        style_idx = random.randint(0,len(self.style_img_files) - 1)
        style_img = Image.open(self.style_img_files[style_idx]).convert('RGB')
        style_img = self.transform(style_img)
        return style_img

    def list_files(self, in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files