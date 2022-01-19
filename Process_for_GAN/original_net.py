import torch
import torch.nn as nn
from torch import optim
import numpy as np
import torch.utils.data as Data
class Original_net(nn.Module):
    def __init__(self):
        super(Original_net, self).__init__()
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
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.colorab = nn.Sequential(
            nn.ConvTranspose2d(256, 2, kernel_size=8, stride=4, padding=2, bias=True),
        )
        
    def forward(self, input):
        conv1_2 = self.conv1(input)
        conv2_2 = self.conv2(conv1_2)
        conv3_3 = self.conv3(conv2_2)
        conv4_3 = self.conv4(conv3_3)
        conv5_3 = self.conv5(conv4_3)
        conv6_3 = self.conv6(conv5_3)
        conv7_3 = self.conv7(conv6_3)
        conv8_3 = self.conv8(conv7_3)
        out_reg = self.colorab(conv8_3)
        
        return out_reg
    
pics = np.load('./pics.npy')
labels_a = np.load('./labels.npy')
torch_dataset = Data.TensorDataset(torch.Tensor(np.asarray(pics)), torch.Tensor(np.asarray(labels_a)))
loader = Data.DataLoader(dataset=torch_dataset, batch_size=50, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Original_net()
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.001)
cnt = 0
for iterate in range(0,20):
    print(iterate)
    for steps, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(data)
        err = criterion(output, label)
        err.backward()
        optimizer.step()
        print(steps)

torch.save(net, './origin.pkl')