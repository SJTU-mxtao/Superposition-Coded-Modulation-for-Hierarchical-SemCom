import torch
from torch import optim, nn
import pandas as pd
from tqdm import tqdm
from eval import EVAL_proposed, EVAL_analog


def train(config, net, train_iter, test_iter, device):
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    results_bad = {'epoch': [], 'loss': [], 'acc': [], 'psnr': [], 'mse': []}
    results_good = {'epoch': [], 'loss': [], 'acc': [], 'psnr': [], 'mse': []}
    resi_track = {'epoch': [], 'residual': []}

    epochs = config.train_iters

    # prob_convs using higher learning rate
    ignored_params = list(map(id, net.basic_encoder.prob_convs.parameters())) + list(map(id, net.enhancement_encoder.prob_convs.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, base_params)},
        {'params': filter(lambda p: p.requires_grad, net.basic_encoder.prob_convs.parameters()), 'lr': config.lr * 60},
        {'params': filter(lambda p: p.requires_grad, net.enhancement_encoder.prob_convs.parameters()), 'lr': config.lr * 60}], config.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs+1, T_mult=1, eta_min=1e-5,
                                                                     last_epoch=-1)
    for epoch in range(epochs):
        total = 0
        acc_total_bad = 0
        acc_total_good_fine = 0
        epoch_loss = []
        net.train()
        for i, (X, Y_coarse, Y_fine) in enumerate(tqdm(train_iter)):
            X, Y_coarse, Y_fine = X.to(device), Y_coarse.to(device), Y_fine.to(device)
            total += len(Y_coarse)
            optimizer.zero_grad()

            u_resi, z, z_hat_bad, z_hat_good, rec_bad, pred_bad, rec_good, pred_good = net(X)
            ce_loss_good = loss_ce(pred_good, Y_fine)
            ce_loss_bad = loss_ce(pred_bad, Y_coarse)

            mse_loss_good = loss_mse(rec_good, X)
            mse_loss_bad = loss_mse(rec_bad, X)
            resi_loss = torch.mean(u_resi ** 2)

            if config.phase == '1':
                loss = ce_loss_bad + config.alpha_1 * mse_loss_bad
            elif config.phase == '2':
                loss = ce_loss_good + config.alpha_2 * mse_loss_good + 10 * resi_loss
            elif config.phase == '3':
                loss = ce_loss_bad + config.alpha_1 * mse_loss_bad \
                       + config.tradeoff * (ce_loss_good + config.alpha_2 * mse_loss_good) + 10 * resi_loss
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.cpu().item())
            acc_train_bad = (pred_bad.data.max(1)[1] == Y_coarse.data).float().sum()
            acc_total_bad += acc_train_bad
            acc_train_good_fine = (pred_good.data.max(1)[1] == Y_fine.data).float().sum()
            acc_total_good_fine += acc_train_good_fine
        scheduler.step()

        loss = sum(epoch_loss) / len(epoch_loss)
        acc_bad, acc_good, psnr_bad, psnr_good, mse_bad, mse_good = EVAL_proposed(net, test_iter, device, config, epoch)
        print('[epoch: {:d}/loss: {:.6f}] \nBad receiver: acc: {:.3f}, psnr: {:.3f}, mse:{:.5f}'
              .format(epoch, loss, acc_bad, psnr_bad, mse_bad))
        print('Good receiver: acc: {:.3f}, psnr: {:.3f}, mse: {:.5f}'
              .format(acc_good, psnr_good, mse_good))

        acc_bad_num = acc_bad.detach().cpu().numpy()
        acc_good_num = acc_good.detach().cpu().numpy()

        results_bad['epoch'].append(epoch)
        results_bad['loss'].append(loss)
        results_bad['acc'].append(acc_bad_num)
        results_bad['psnr'].append(psnr_bad)
        results_bad['mse'].append(mse_bad)

        results_good['epoch'].append(epoch)
        results_good['loss'].append(loss)
        results_good['acc'].append(acc_good_num)
        results_good['psnr'].append(psnr_good)
        results_good['mse'].append(mse_good)

        resi_track['epoch'].append(epoch)
        resi_track['residual'].append(resi_loss)

    # save trained models
    if config.phase == '3':
        model_name = '/{}_SNR{:.1f}_{:.1f}_Trans{:d}_{}_phase{}_{}_{}.pth.tar'.format(
            config.net,
            config.snr_train_good,
            config.snr_train_bad,
            config.channel_use,
            config.sp_mode,
            config.phase,
            config.a,
            config.tradeoff
        )
    else:
        model_name = '/{}_SNR{:.1f}_{:.1f}_Trans{:d}_{}_phase{}_{}.pth.tar'.format(
            config.net,
            config.snr_train_good,
            config.snr_train_bad,
            config.channel_use,
            config.sp_mode,
            config.phase,
            config.a
        )
    torch.save(net.state_dict(), config.model_path + model_name)
    print('Successfully save present model!')

    # save training data
    file_name_bad = '/{}_BADSNR{:.1f}_GOODSNR{:.1f}_Trans{:d}_Bad_{}_phase{}_{}.csv'.format(
        config.net,
        config.snr_train_bad,
        config.snr_train_good,
        config.channel_use,
        config.sp_mode,
        config.phase,
        config.a,
    )
    file_name_good = '/{}_BADSNR{:.1f}_GOODSNR{:.1f}_Trans{:d}_Good_{}_phase{}_{}.csv'.format(
        config.net,
        config.snr_train_bad,
        config.snr_train_good,
        config.channel_use,
        config.sp_mode,
        config.phase,
        config.a,
    )
    data = pd.DataFrame(results_bad)
    data.to_csv(config.result_path + file_name_bad, index=False, header=False)

    data = pd.DataFrame(results_good)
    data.to_csv(config.result_path + file_name_good, index=False, header=False)


def train_analog(config, net, train_iter, test_iter, device):
    epochs = config.train_iters

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=1e-4)
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs + 1, T_mult=1, eta_min=1e-6,
                                                                     last_epoch=-1)

    results = {'epoch': [], 'loss': [], 'train_acc': [], 'acc': [], 'psnr': []}

    for epoch in range(epochs):
        total = 0
        acc_total = 0
        mse_total = 0
        net.train()
        epoch_loss = []
        for i, (X, Y_coarse, Y_fine) in enumerate(tqdm(train_iter)):
            X, Y_coarse, Y_fine = X.to(device), Y_coarse.to(device), Y_fine.to(device)

            total += len(Y_coarse)
            optimizer.zero_grad()
            rec, pred = net(X)
            if config.net == 'analog_bad':
                loss = loss_ce(pred, Y_coarse) + config.alpha_1 * loss_mse(rec, X)
            elif config.net == 'analog_good':
                loss = loss_ce(pred, Y_fine) + config.alpha_1 * loss_mse(rec, X)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().item())
            if config.net == 'analog_bad':
                acc_train = (pred.data.max(1)[1] == Y_coarse.data).float().sum()
            elif config.net == 'analog_good':
                acc_train = (pred.data.max(1)[1] == Y_fine.data).float().sum()
            mse = loss_mse((rec + 1) / 2, (X + 1) / 2)
            mse_total += mse
            acc_total += acc_train
        scheduler.step()

        loss = sum(epoch_loss) / len(epoch_loss)
        acc, mse, psnr = EVAL_analog(net, test_iter, device, config)
        print('[epoch: {:d}/loss: {:.6f}] \nReceiver, ACC: {:.3f}; MSE: {:.6}; PSNR: {:.3f}'
              .format(epoch, loss, acc, mse, psnr))

        acc_num = acc.detach().cpu().numpy()
        results['epoch'].append(epoch)
        results['loss'].append(loss)
        results['train_acc'].append((acc_total / total).cpu().numpy())
        results['acc'].append(acc_num)
        results['psnr'].append(psnr)

    model_name = '/{}_model_Trans{:d}_snr{}.pth.tar'.format(
        config.net,
        config.channel_use,
        config.snr_train_good,
    )
    torch.save(net.state_dict(), config.model_path + model_name)
    print('Successfully save present model!')

    file_name_bad = '/{}_Trans{:d}_snr{}.csv'.format(
        config.net,
        config.channel_use,
        config.snr_train_good,
    )
    data = pd.DataFrame(results)
    data.to_csv(config.result_path + file_name_bad, index=False, header=False)
