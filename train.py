import time
import argparse

from torch import utils
from datasets.yuvdatasets import yuvdatasets

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from models.Decoder import Decoder
from models.Encoder import Encoder
from utils import processdata
import os
from utils import processdata
from torch.utils.tensorboard import SummaryWriter
import tensorboard


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN based model")
    parser.add_argument("-d",
                        "--training_dir",
                        type=str,
                        required=True,
                        help="Training directory (folder contains all 256*256 images)")
    parser.add_argument("-t",
                        "--test_image",
                        type=str,
                        default=None,
                        help="Test image location")
    parser.add_argument("-c",
                        "--checkpoint_location",
                        type=str,
                        default="./checkpoints",
                        help="Place to save checkpoints")
    parser.add_argument("-e",
                        "--epoch",
                        type=int,
                        default=120,
                        help="Epoches to run training")
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=24,
                        help="batch size")
    parser.add_argument("-w",
                        "--num_workers",
                        type=int,
                        default=6,
                        help="Number of workers to fetch data")
    parser.add_argument("-p",
                        "--pixel_loss_weights",
                        type=float,
                        default=1000.0,
                        help="Pixel-wise loss weights")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("-i",
                        "--checkpoint_every",
                        type=int,
                        default=500,
                        help="Save checkpoint every k iteration (checkpoints for same epoch will overwrite)")
    parser.add_argument("-log",
                        "--log_every",
                        type=int,
                        default=50,
                        help="show the image every k iteration")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    device = torch.device("cuda")
    dataset = yuvdatasets(cont_img_path=opt.training_dir, style_img_path=opt.training_dir, img_size=256)
    dataset_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers), drop_last=True)

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    encoder_backbone = torchvision.models.mobilenet_v3_large(pretrained=True).to(device)
    encoder = Encoder(encoder_backbone, net_type="mobilenetv3").to(device)
    encoder.eval()

    decoder = Decoder()
    decoder = decoder.to(device)
    decoder.train()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    writer = SummaryWriter()
    batch_done = 0
    for epoch in range(opt.epoch):
        # for i, data in enumerate(dataset):
        for i, (rgb) in enumerate(dataset_loader):
            rgb = rgb.to(device)
            yuv = processdata.rgb2yuv_601(rgb)
            optimizer.zero_grad()
            gray = yuv[:, 0:1].repeat(1, 3, 1, 1)
            encoder_output = encoder.encode_with_intermediate(gray)
            output = decoder(encoder_output[-1], encoder_output[-2], encoder_output[-3], encoder_output[-4], encoder_output[-5])
            output_rgb = processdata.yuv2rgb_601(yuv[:, 0:1], output)
            loss = encoder.content_loss(rgb, output_rgb)
            loss.backward()
            writer.add_scalar("loss", loss.item(), batch_done)
            batch_done += 1
            if batch_done % opt.log_every == 0:
                gray = make_grid(gray, nrow=opt.batch_size, normalize=False)
                color = make_grid(rgb, nrow=opt.batch_size, normalize=False)
                out = make_grid(output_rgb, nrow=opt.batch_size, normalize=False)

                image_grid = torch.cat((gray, color, out), 1)
                writer.add_image("log_picture", image_grid, batch_done)
            
            if batch_done % opt.checkpoint_every == 0:
                ckpt_model_filename = "ckpt_" + str(epoch) + '_' + str(batch_done) + ".pth"
                ckpt_model_path = os.path.join(opt.checkpoint_location, ckpt_model_filename)
                torch.save(decoder.state_dict(), ckpt_model_path)
            optimizer.step()
