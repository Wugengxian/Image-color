import time
import argparse

from torch import utils
from datasets import imgDataset

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from models import ECCV6
from util import processdata
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
                        required=True,
                        help="Place to save checkpoints")
    parser.add_argument("-e",
                        "--epoch",
                        type=int,
                        default=120,
                        help="Epoches to run training")
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=20,
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
                        default=100,
                        help="Save checkpoint every k iteration (checkpoints for same epoch will overwrite)")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    dataset = imgDataset(opt.training_dir)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers))

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = ECCV6.eccv16()
    model = model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    loss_fuction = nn.MSELoss()

    for epoch in range(opt.epoch):
        # for i, data in enumerate(dataset):
        for i, (tens_orig, tens_rs_l, tens_rs_ab) in enumerate(dataset_loader):
            optimizer.zero_grad()
            tens_rs = tens_rs.cuda()
            output = model(tens_rs_l)
            loss = loss_fuction(tens_rs_ab, output)
            loss.backward()
            optimizer.step()
