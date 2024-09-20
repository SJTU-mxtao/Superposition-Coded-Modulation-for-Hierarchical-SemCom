import torch
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader as DataLoader
from dataset import Cifar100
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim
from skimage.metrics import mean_squared_error as comp_mse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import random


def data_loader(data_root, batch_size, transform_train, transform_test):

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    cifar100_training = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=None,
        download=True,
    )
    train_set = Cifar100(cifar100_training, transform_train)
    train_loader = DataLoader(train_set, batch_size, True)

    cifar100_testing = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        transform=None,
        download=True,
    )
    test_set = Cifar100(cifar100_testing, transform_test)
    test_loader = DataLoader(test_set, batch_size, True)

    return train_loader, test_loader

    
def init_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 42:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def denorm(x, channels=None, w=None, h=None, resize=False):
    x = (x * 0.5 + 0.5).clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x


def PSNR(tensor_org, tensor_trans):
    total_psnr = 0
    tensor_org = (tensor_org + 1) / 2
    tensor_trans = (tensor_trans + 1) / 2
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        psnr = 0
        for j in range(np.size(trans, 1)):
            psnr_temp = comp_psnr(origin[i, j, :, :], trans[i, j, :, :])
            psnr = psnr + psnr_temp
        psnr /= 3
        total_psnr += psnr

    return total_psnr


def SSIM(tensor_org, tensor_trans):
    total_ssim = 0
    tensor_org = (tensor_org + 1) / 2
    tensor_trans = (tensor_trans + 1) / 2
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        ssim = comp_ssim(origin[i, :, :, :], trans[i, :, :, :], channel_axis=0)
        total_ssim += ssim

    return total_ssim


def MSE(tensor_org, tensor_trans):
    total_mse = 0
    tensor_org = (tensor_org + 1) / 2
    tensor_trans = (tensor_trans + 1) / 2
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        mse = 0
        for j in range(np.size(trans, 1)):
            mse_temp = comp_mse(origin[i, j, :, :], trans[i, j, :, :])
            mse = mse + mse_temp
        mse /= 3
        total_mse += mse

    return total_mse


def count_percentage_super(code, mod, epoch, snr, channel_use, tradeoff_h, name, phase, a, tradeoff):

    code = code.reshape(-1)
    index = [i for i in range(len(code))]
    random.shuffle(index)
    code = code[index]
    code = code.reshape(-1, 2).cpu()

    I_point = torch.unique(code.reshape(-1))

    if mod == '4and4':
        order = 16
    elif mod == '4and16':
        order = 64

    I, Q = torch.meshgrid(I_point, I_point)
    map = torch.cat((I.unsqueeze(-1), Q.unsqueeze(-1)), dim=2).reshape(order, 2)
    per_s = []
    fig = plt.figure(dpi=300)
    ax = Axes3D(fig)
    fig.add_axes(ax)
    for i in range(order):
        temp = torch.sum(torch.abs(code - map[i, :]), dim=1)
        num = code.shape[0] - torch.count_nonzero(temp).item()
        per = num / code.shape[0]
        per_s.append(per)

    per_s = torch.tensor(per_s).cpu()
    height = np.zeros_like(per_s)
    width = depth = 0.05
    surf = ax.bar3d(I.ravel(), Q.ravel(), height, width, depth, per_s, zsort='average', alpha=0.6, color='lightcoral')
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    if phase == '3':
        file_name = './cons_fig/' + '{}_{}_{}_{}_{}_phase{}_{}_{}'.format(name, mod, snr, channel_use, tradeoff_h, phase, a, tradeoff)
    else:
        file_name = './cons_fig/' + '{}_{}_{}_{}_{}_phase{}_{}'.format(name, mod, snr, channel_use, tradeoff_h, phase,
                                                                       a)
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    fig.savefig(file_name + '/{}'.format(epoch))
    plt.close()

    fig = plt.figure(dpi=300)
    for k in range(order):
        plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='lightcoral')
    fig.savefig(file_name + '/scatter_{}'.format(epoch))
    plt.close()