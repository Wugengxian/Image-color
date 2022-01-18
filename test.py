import time
import argparse
from matplotlib.pyplot import style

from torch import utils
from datasets.yuvdatasets import yuvdatasets

import torch
from torch import nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from models.Decoder import Decoder
from models.Encoder import Encoder
from models.Discriminator import NLayerDiscriminator
from models.generator import ResUnetPlusPlus, UnetGenerator
from loss.loss import GDLLoss, GANLoss
from utils import processdata
import os
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
    parser.add_argument("-load",
                        "--state_dict",
                        type=str,
                        default=None,
                        help="Place to load checkpoints")
    parser.add_argument("-e",
                        "--epoch",
                        type=int,
                        default=60,
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
    parser.add_argument("-loss",
                        "--loss_type",
                        type=str,
                        default="0",
                        help="use the loss layer of the mobilenetv3")
    parser.add_argument("--preference",
                        action='store_true',
                        help="consider to use preference image")
    parser.add_argument("--gan",
                        action='store_true',
                        help="consider to use GAN")
    parser.add_argument("--gan_g",
                        action='store_true',
                        help="consider to use GAN")
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
    loss_grad = GDLLoss()
    loss_l1 = nn.L1Loss()
    encoder.eval()

    if opt.gan or opt.gan_g:
        D = NLayerDiscriminator(3).to(device)
        optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        loss_GAN = GANLoss(gan_mode='lsgan').to(device)
        D.eval()

    if opt.gan_g:
        decoder = UnetGenerator(1, 2, 8)
    else:
        decoder = Decoder()
    decoder = decoder.to(device)
    if opt.state_dict is not None:
        decoder.load_state_dict(torch.load(opt.state_dict))
    decoder.eval()


    writer = SummaryWriter()
    batch_done = 0
    for epoch in range(opt.epoch):
        # for i, data in enumerate(dataset):
        for i, (rgb) in enumerate(dataset_loader):
            rgb = rgb.to(device)
            yuv = processdata.rgb2yuv_601(rgb)
            gray = yuv[:, 0:1]
            gray_3 = gray.repeat(1, 3, 1, 1)
            if opt.preference:
                style_img = dataset.getStyle().to(device).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1)
                style_feat = encoder.encode_with_intermediate(style_img)
                encoder_output = encoder.encode_with_instancenormal(gray_3, style_feat)
            else:
                encoder_output = encoder.encode_with_intermediate(gray_3)
            if opt.gan_g:
                output = decoder(gray)
            else:
                output = decoder(encoder_output[-1], encoder_output[-2], encoder_output[-3], encoder_output[-4], encoder_output[-5])
            output_rgb = processdata.yuv2rgb_601(gray, output)
            if opt.preference:
                if opt.gan:
                    #this is for Discriminator
                    optimizer_D.zero_grad()
                    pred_fake = loss_GAN(D(torch.cat([gray, output], dim=1).detach()), False)
                    pred_true = loss_GAN(D(torch.cat([gray, yuv[:, 1:]], dim=1)), True)
                    loss_D = (pred_fake + pred_true) / 2

                    #this is for generater
                    D.requires_grad(False)
                    loss_c, loss_s = encoder.loss(output_rgb, content = rgb, style_feat=style_feat)
                    loss_g = loss_grad(output_rgb, rgb)
                    loss_gan = loss_GAN(D(torch.cat([gray, output], dim=1)), True)
                    loss = loss_gan + 0.05 * loss_c + 0.05 * loss_s + 0.5 * loss_g

                    writer.add_scalars("loss_G", {"loss_s":loss_s.item(),"loss_c":loss_c.item(), "loss_g": loss_g.item(), 
                                        "loss_G_true": loss_gan.item(), "loss_G": loss.item()}, batch_done)
                    writer.add_scalars("loss_D", {"pred_fake": pred_fake.item(), "pred_true": pred_true.item(), "loss_D": loss_D.item()}, batch_done)
                else:
                    loss_c, loss_s = encoder.loss(output_rgb, content = rgb, style_feat=style_feat)
                    loss_g = loss_grad(output_rgb, rgb)
                    loss = 1 * loss_c + 1 * loss_s + 10 * loss_g
                    writer.add_scalars("loss", {"loss_s":loss_s.item(),"loss_c":loss_c.item(),"loss_g": loss_g.item(),"total_loss":loss.item()}, batch_done)
            else:
                if opt.gan_g:

                    pred_fake = loss_GAN(D(torch.cat([gray, output], dim=1).detach()), False)
                    pred_true = loss_GAN(D(torch.cat([gray, yuv[:, 1:]], dim=1)), True)
                    loss_D = (pred_fake + pred_true) / 2

                    loss_c = loss_l1(output_rgb, rgb)
                    loss_g = loss_grad(output_rgb, rgb)
                    loss_gan = loss_GAN(D(torch.cat([gray, output], dim=1)), True)
                    loss = loss_gan + 2 * loss_c + 5 * loss_g

                    writer.add_scalars("loss_G", {"loss_c":loss_c.item(), "loss_g": loss_g.item(), 
                                        "loss_G_true": loss_gan.item(), "loss_G": loss.item()}, batch_done)
                    writer.add_scalars("loss_D", {"pred_fake": pred_fake.item(), "pred_true": pred_true.item(), "loss_D": loss_D.item()}, batch_done)
                else:
                    loss = encoder.content_loss(rgb, output_rgb, loss_type=opt.loss_type)s
                    writer.add_scalar("loss", loss.item(), batch_done)
            batch_done += 1

            if batch_done % opt.log_every == 0:
                gray_3 = make_grid(gray_3, nrow=opt.batch_size, normalize=False)
                color = make_grid(rgb, nrow=opt.batch_size, normalize=False)
                out = make_grid(output_rgb, nrow=opt.batch_size, normalize=False)

                if opt.preference:
                    style_color = make_grid(style_img, nrow=opt.batch_size, normalize=False)
                    image_grid = torch.cat((gray_3, color, style_color, out), 1)
                else:
                    image_grid = torch.cat((gray_3, color, out), 1)
                writer.add_image("log_picture", image_grid, batch_done)
            
            if batch_done % opt.checkpoint_every == 0:
                ckpt_model_filename = "ckpt_" + str(epoch) + '_' + str(batch_done) + ".pth"
                ckpt_model_path = os.path.join(opt.checkpoint_location, ckpt_model_filename)
                torch.save(decoder.state_dict(), ckpt_model_path)
                if opt.gan or opt.gan_g:
                    ckpt_model_filename = "discriminator_" + str(epoch) + '_' + str(batch_done) + ".pth"
                    ckpt_model_path = os.path.join(opt.checkpoint_location, ckpt_model_filename)
                    torch.save(D.state_dict(), ckpt_model_path)

    
