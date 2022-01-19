import torch
import torch.nn as nn
from torch import optim
import numpy as np
import torch.utils.data as Data

class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.colorab = nn.Sequential(
            nn.ConvTranspose2d(384, 2, kernel_size=8, stride=4, padding=2, bias=True),
        )
        
    def forward(self, input):
        conv1_2 = self.conv1(input)
        conv2_2 = self.conv2(conv1_2)
        conv3_3 = self.conv3(conv2_2)
        conv4_3 = self.conv4(conv3_3)
        conv5_3 = self.conv5(conv4_3)
        conv6_3 = self.conv6(conv5_3)
        conv7_3 = self.conv7(conv6_3)
        conv8_3 = self.conv8(torch.cat((conv3_3, conv7_3),dim=1))
        out_reg = self.colorab(torch.cat((conv2_2,conv8_3),dim=1))
        
        return out_reg

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.maxpool1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.maxpool2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.maxpool3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.maxpool4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.maxpool5= nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.dense = nn.Sequential(
            nn.Linear(32768, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2)
        )
 
    def forward(self, x):
        pool1=self.maxpool1(x)
        pool2=self.maxpool2(pool1)
        pool3=self.maxpool3(pool2)
        pool4=self.maxpool4(pool3)
        pool5=self.maxpool5(pool4)
        
        flat = pool5.view(pool5.size(0), -1)
        class_ = self.dense(flat)
        return class_

class Discriminator_ab2l(nn.Module):
    def __init__(self):
        super(Discriminator_ab2l, self).__init__()
        self.maxpool1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.maxpool2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.maxpool3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.maxpool4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.maxpool5= nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.dense = nn.Sequential(
            nn.Linear(32768, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2)
        )
 
    def forward(self, x):
        pool1=self.maxpool1(x)
        pool2=self.maxpool2(pool1)
        pool3=self.maxpool3(pool2)
        pool4=self.maxpool4(pool3)
        pool5=self.maxpool5(pool4)
        
        flat = pool5.view(pool5.size(0), -1)
        class_ = self.dense(flat)
        return class_
    
class U_net_ab2l(nn.Module):
    def __init__(self):
        super(U_net_ab2l, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.colorab = nn.Sequential(
            nn.ConvTranspose2d(384, 1, kernel_size=8, stride=4, padding=2, bias=True),
        )
        
    def forward(self, input):
        conv1_2 = self.conv1(input)
        conv2_2 = self.conv2(conv1_2)
        conv3_3 = self.conv3(conv2_2)
        conv4_3 = self.conv4(conv3_3)
        conv5_3 = self.conv5(conv4_3)
        conv6_3 = self.conv6(conv5_3)
        conv7_3 = self.conv7(conv6_3)
        conv8_3 = self.conv8(torch.cat((conv3_3, conv7_3),dim=1))
        out_reg = self.colorab(torch.cat((conv2_2,conv8_3),dim=1))
        
        return out_reg

pics = np.load('./pics.npy')
labels_a = np.load('./labels.npy')
torch_dataset = Data.TensorDataset(torch.Tensor(np.asarray(pics)), torch.Tensor(np.asarray(labels_a)))
loader = Data.DataLoader(dataset=torch_dataset, batch_size=10, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator_l2ab = torch.load('./generator_l2ab.pkl')
generator_ab2l = torch.load('./generator_ab2l.pkl')
generator_l2ab.to(device)
generator_ab2l.to(device)
discriminator_l2ab = torch.load('./discriminator_l2ab.pkl')
discriminator_ab2l = torch.load('./discriminator_ab2l.pkl')
discriminator_l2ab.to(device)
discriminator_ab2l.to(device)
criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
optimizerDl2ab = optim.RMSprop(discriminator_l2ab.parameters(), lr=0.0007)
optimizerGl2ab = optim.Adam(generator_l2ab.parameters(), lr=0.001)
optimizerDab2l = optim.RMSprop(discriminator_ab2l.parameters(), lr=0.0007)
optimizerGab2l = optim.Adam(generator_ab2l.parameters(), lr=0.001)

cnt = 0
for iterate in range(0,5):
    print("start:",str(iterate))
    for steps, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        optimizerDl2ab.zero_grad()
        fake_ab = generator_l2ab(data)
        fake = torch.cat((data,fake_ab),dim=1)
        real_img = torch.cat((data,label), dim=1)
        outputDr = discriminator_l2ab(real_img)
        outputDf = discriminator_l2ab(fake.detach())
        # batch size == 25
        real_label = [1]*10
        real_label = torch.LongTensor(real_label).to(device)
        false_label = [0]*10
        false_label = torch.LongTensor(false_label).to(device)
        
        errReal = criterion(outputDr, real_label)
        errFalse = criterion(outputDf, false_label)
        errD = (errFalse+errReal)/2
        errD.backward()
        optimizerDl2ab.step()
        
        # l2ab G
        optimizerGl2ab.zero_grad()
        outputG = discriminator_l2ab(fake.detach())
        errG = criterion(outputG,real_label)
        outputG2 = generator_ab2l(fake_ab.detach())
        errG2 = criterion2(outputG2,data)
        errG3 = criterion2(fake_ab, label)
        errGs = (errG + 30*errG2 + 10*errG3)
        errGs.backward()
        optimizerGl2ab.step()
        
        # ab2l D
        optimizerDab2l.zero_grad()
        fake_l = generator_ab2l(label)
        fake2 = torch.cat((fake_l,label),dim=1)
        real_img2 = torch.cat((data,label),dim=1)
        outputDr2 = discriminator_ab2l(real_img2)
        outputDf2 = discriminator_ab2l(fake2.detach())
        
        real_label2 = [1]*10
        real_label2 = torch.LongTensor(real_label2).to(device)
        false_label2 = [0]*10
        false_label2 = torch.LongTensor(false_label2).to(device)
        
        errReal2 = criterion(outputDr2, real_label2)
        errFalse2 = criterion(outputDf2, false_label2)
        errD2 = (errFalse2+errReal2)/2
        errD2.backward()
        optimizerDab2l.step()
        
        # ab2l G
        optimizerGab2l.zero_grad()
        outputG = discriminator_ab2l(fake2.detach())
        errG = criterion(outputG,real_label)
        outputG2 = generator_l2ab(fake_l.detach())
        errG2 = criterion2(outputG2,label)
        errG3 = criterion2(fake_l,data)
        errGs = (errG + 30*errG2 + 10*errG3)
        errGs.backward()
        optimizerGab2l.step()
        
        print(iterate,"loss D:",errD,"loss G:",errG)

torch.save(generator_ab2l, './generator_ab2l.pkl')
torch.save(discriminator_ab2l, './discriminator_ab2l.pkl')
torch.save(generator_l2ab, './generator_l2ab.pkl')
torch.save(discriminator_l2ab, './discriminator_l2ab.pkl')