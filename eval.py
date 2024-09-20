from utils import MSE, PSNR
import torch
from tqdm import tqdm
from utils import count_percentage_super
from torch import optim, nn
import math


def EVAL_proposed(model, data_loader, device, config, epoch):
    model.eval()
    acc_total_bad = 0
    acc_total_good = 0
    psnr_total_bad = 0
    psnr_total_good = 0
    mse_total_bad = 0
    mse_total_good = 0
    total = 0
    for batch_idx, (data, target_coarse, target_fine) in enumerate(tqdm(data_loader)):
        data, target_coarse, target_fine = data.to(device), target_coarse.to(device), target_fine.to(device)
        total += len(target_coarse)
        with torch.no_grad():
            _, z, z_hat_bad, z_hat_good, rec_bad, pred_bad, rec_good, pred_good = model(data)

        if batch_idx == 0:
            count_percentage_super(z, config.sp_mode, epoch, config.snr_train_good, config.channel_use, config.alpha_1,
                             config.net, config.phase, config.a, config.tradeoff)
        acc_bad = (pred_bad.data.max(1)[1] == target_coarse.data).float().sum()
        acc_good = (pred_good.data.max(1)[1] == target_fine.data).float().sum()

        psnr_bad = PSNR(data, rec_bad)
        psnr_good = PSNR(data, rec_good)

        mse_bad = MSE(data, rec_bad)
        mse_good = MSE(data, rec_good)

        acc_total_bad += acc_bad
        acc_total_good += acc_good
        psnr_total_bad += psnr_bad
        psnr_total_good += psnr_good
        mse_total_bad += mse_bad
        mse_total_good += mse_good

    acc_total_bad /= total
    acc_total_good /= total
    psnr_total_bad /= total
    psnr_total_good /= total
    mse_total_bad /= total
    mse_total_good /= total

    return acc_total_bad, acc_total_good, psnr_total_bad, psnr_total_good, mse_total_bad, mse_total_good


def EVAL_analog(model, data_loader, device, config):
    model.eval()
    acc_total = 0
    mse_total = 0
    total = 0
    loss_mse = nn.MSELoss()
    for batch_idx, (data, target_coarse, target_fine) in enumerate(tqdm(data_loader)):
        data, target_coarse, target_fine = data.to(device), target_coarse.to(device), target_fine.to(device)
        total += len(target_fine)
        with torch.no_grad():
            rec, pred = model(data)
        if config.net == 'analog_bad':
            acc = (pred.data.max(1)[1] == target_coarse.data).float().sum()
        elif config.net == 'analog_good':
            acc = (pred.data.max(1)[1] == target_fine.data).float().sum()
        loss = loss_mse((rec + 1) / 2, (data + 1) / 2)
        acc_total += acc
        mse_total += loss

    acc_total /= total
    mse_total /= (batch_idx + 1)
    psnr = 10 * math.log(1 / mse_total, 10)

    return acc_total, mse_total, psnr