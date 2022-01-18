from turtle import forward
import torch
import torch.nn as nn
from models.layers import *
from utils.processdata import *

class Encoder(nn.Module):
    def __init__(self, encoder, net_type='vgg', fixed=True):
        super(Encoder, self).__init__()
        self.net_type = net_type
        if net_type == 'vgg':
            enc_layers = list(encoder.children())
            self.enc = []
            self.enc.append(nn.Sequential(*enc_layers[:4]))
            self.enc.append(nn.Sequential(*enc_layers[4:11]))
            self.enc.append(nn.Sequential(*enc_layers[11:18]))
            self.enc.append(nn.Sequential(*enc_layers[18:31]))
        elif net_type == 'mobilenetv3':
            enc_layers = list(encoder.features.children())
            self.enc = []
            device = torch.device("cuda")
            self.enc_1 = nn.Sequential(Normalization(normalization_mean, normalization_std).to(device), *enc_layers[:2])
            self.enc_2 = nn.Sequential(*enc_layers[2:4])
            self.enc_3 = nn.Sequential(*enc_layers[4:7])
            self.enc_4 = nn.Sequential(*enc_layers[7:13])
            self.enc_5 = nn.Sequential(*enc_layers[13:16])
        else:
            raise ValueError("please enter a net type")
        self.mse_loss = nn.MSELoss()

        if fixed:
            for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
                for param in getattr(self, name).parameters():
                    param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def encode_with_instancenormal(self, input, output, alpha=1):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(adaptive_instance_normalization(func(results[-1]), output[i+1], alpha=alpha))
        return results

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            input = func(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)  # swapped ch and w*h, transpose share storage with original
        gram = features.bmm(features_t) / (ch * h * w)
        return gram
    
    def content_loss(self, input1, input2, loss_type="5"):
        cal_layer = loss_type.split(",")
        cal_layer = [int(num) for num in cal_layer]
        out1 = self.encode_with_intermediate(input1)
        out2 = self.encode_with_intermediate(input2)
        loss = None
        for layer in cal_layer:
            if loss is None:
                loss = self.mse_loss(out1[layer], out2[layer])
            else:
                loss += self.mse_loss(out1[layer], out2[layer])
        return loss

    def loss(self,output, content=None, style=None, style_feat=None, cont_feat=None):
        if style_feat is None:
            style_feats = self.encode_with_intermediate(style)
        else:
            style_feats = style_feat
        if cont_feat is None:
            content_feat = self.encode(content)
        else:
            content_feat = cont_feat

        out_feats = self.encode_with_intermediate(output)

        loss_c = self.calc_content_loss(out_feats[-1], content_feat)
        loss_s = self.calc_style_loss(out_feats[1], style_feats[1])
        for i in range(4):
            loss_s += self.calc_style_loss(out_feats[i+1], style_feats[i+1])
        return loss_c, loss_s
