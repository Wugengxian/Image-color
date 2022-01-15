from turtle import forward
import torch
import torch.nn as nn
from models.layers import *

class Encoder(nn.Module):
    def __init__(self, encoder, net_type='vgg'):
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
            self.enc.append(nn.Sequential(Normalization(normalization_mean, normalization_std).to(device), *enc_layers[:2]))
            self.enc.append(nn.Sequential(*enc_layers[2:4]))
            self.enc.append(nn.Sequential(*enc_layers[4:7]))
            self.enc.append(nn.Sequential(*enc_layers[7:13]))
            self.enc.append(nn.Sequential(*enc_layers[13:16]))
        else:
            raise ValueError("please enter a net type")
        self.mse_loss = nn.MSELoss()

        for i in self.enc:
            for param in i.parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for func in self.enc:
            results.append(func(results[-1]))
        return results[1:]
    
    # def encode_with_instancenormal(self, input, output, alpha=1, use_WCT=False):
    #     results = [input]
    #     for i in range(len(self.enc)):
    #         func = self.enc[i]
    #         if use_WCT:
    #             results.append(wct_core(func(results[-1]), output[i], alpha=alpha))
    #         else:
    #             results.append(adaptive_instance_normalization(func(results[-1]), output[i], alpha=alpha))
    #     return results[1:]        

    # extract relu4_1 from input image
    def encode(self, input):
        for func in self.enc:
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
    
    def content_loss(self, input1, input2):
        out1 = self.encode(input1)
        out2 = self.encode(input2)
        return self.mse_loss(out1, out2)

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
        # loss_s = torch.tensor(0.0).cuda()
        loss_s = 0.0
        for i in range(len(self.enc)):
            O = self.gram_matrix(out_feats[i])
            S = self.gram_matrix(style_feats[i])
            loss_s += self.mse_loss(O, S)
        return loss_c, loss_s
