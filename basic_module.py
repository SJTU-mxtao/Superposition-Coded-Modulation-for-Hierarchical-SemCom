from torch import nn
import torch
from torch.nn.functional import gumbel_softmax
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torchvision import models


def modulation(probs, order, device, p=1):
    eps = 1e-10
    # n, l = probs.shape

    order_sqrt = int(order ** 0.5)
    prob_z = gumbel_softmax(probs, hard=False)
    discrete_code = gumbel_softmax(probs, hard=True, tau=1)

    if order_sqrt == 2:
        const = [1, -1]
    elif order_sqrt == 4:
        const = [-3, -1, 1, 3]
    elif order_sqrt == 8:
        const = [-7, -5, -3, -1, 1, 3, 5, 7]

    const = torch.tensor(const).to(device)
    temp = discrete_code * const
    output = torch.sum(temp, dim=2)

    return output, prob_z


class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)
        return x


def awgn(snr, x, device, p=1):
    # snr(db)
    n = p / (10 ** (snr / 10))
    sqrt_n = n ** 0.5
    noise = torch.randn_like(x) * sqrt_n
    noise = noise.to(device)
    x_hat = x + noise
    return x_hat


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.conv_init_state_dict = list(net.children())[0].state_dict()
        self.conv1[0].load_state_dict(self.conv_init_state_dict)
        self.inchannel = 512
        self.layer1 = nn.Sequential(*list(net.children())[4:6])
        self.layer2 = self.make_layer(ResidualBlock, 512, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 512, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, config.channel_use * 2, 2, stride=2)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Decoder_Recon(nn.Module):
    def __init__(self, config):
        super(Decoder_Recon, self).__init__()

        params = [512, 32, 256, 64]

        input_channel = int(config.channel_use * 2 / (4 * 4))

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, params[0], 1, 1, 0),
            nn.BatchNorm2d(params[0]),
            nn.ReLU())

        self.inchannel = params[0]

        self.layer1 = self.make_layer(ResidualBlock, params[0], 2, stride=1)

        self.layer2 = self.make_layer(ResidualBlock, params[0], 2, stride=1)

        self.DepthToSpace1 = DepthToSpace(4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(params[1], params[2], 1, 1, 0),
            nn.BatchNorm2d(params[2]),
            nn.ReLU())

        self.inchannel = params[2]

        self.layer3 = self.make_layer(ResidualBlock, params[2], 2, stride=1)

        self.layer4 = self.make_layer(ResidualBlock, params[2], 2, stride=1)

        self.DepthToSpace2 = DepthToSpace(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(params[3], 3, 1, 1, 0),
            nn.BatchNorm2d(3))

        self.sigmoid = nn.Sigmoid()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, z):
        z0 = self.conv1(z)
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.DepthToSpace1(z2)
        z4 = self.conv2(z3)
        z5 = self.layer3(z4)
        z5 = self.layer4(z5)
        z5 = self.DepthToSpace2(z5)
        z6 = self.conv3(z5)
        z6 = (self.sigmoid(z6) - 0.5) * 2
        return z6


class Decoder_Class(nn.Module):
    def __init__(self, half_width, layer_width, num_class):
        super(Decoder_Class, self).__init__()
        self.layer_width = half_width
        self.Half_width = layer_width
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width, self.layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(self.layer_width * 4, num_class)

    def forward(self, z):
        x1 = self.fc_spinal_layer1(z[:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([z[:, self.Half_width:2 * self.Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([z[:, 0:self.Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([z[:, self.Half_width:2 * self.Half_width], x3], dim=1))
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        pred = self.last_fc(x)
        return pred


def normalize(x, power=1):
    power_emp = torch.mean(torch.abs(x) ** 2)
    x = (power / power_emp) ** 0.5 * x
    return power_emp, x
