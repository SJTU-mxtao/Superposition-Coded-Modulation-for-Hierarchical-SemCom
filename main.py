import torch
import torchvision.transforms as transforms
from scm_net import SuperpositionNet
from train import train, train_analog
from utils import init_seeds, data_loader
import os
import argparse
from eval import EVAL_proposed
from pretrain_net import Analog


def mischandler(config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.rec_path):
        os.makedirs(config.rec_path)
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)


def main(config):
    # set seeds
    init_seeds()

    # prepare data
    transform_train = [
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomVerticalFlip(p=0.5),
                       transforms.RandomResizedCrop(32, scale=(0.8, 1)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                       ]

    transform_test = [transforms.ToTensor(),
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                      ]

    train_loader, test_loader = data_loader(config.dataset_path, config.batch_size, transform_train, transform_test)

    if config.mode == 'train':
        start_train(config, train_loader, test_loader)
    elif config.mode == 'test':
        start_test(config, test_loader)


def start_train(config, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.net == 'scm':
        print('Superposition is {}.'.format(config.sp_mode))
        print('Channel use is: {}.'.format(config.channel_use))
        print('The SNR of the good user: {}.'.format(config.snr_train_good))
        print('a = {}.'.format(config.a))

        """Loading pretrained analog model"""
        config.net = 'analog_good'
        pretrain_net = Analog(config, device).to(device)
        model_name = '/analog_good_model_Trans{:d}_snr20.pth.tar'.format(
            config.channel_use,
        )
        pretrain_net.load_state_dict(torch.load(config.model_path + model_name))
        encoder_enhancement_dict = pretrain_net.encoder.state_dict()
        rec_decoder_good_dict = pretrain_net.rec_decoder.state_dict()
        class_decoder_good_dict = pretrain_net.class_decoder.state_dict()

        config.net = 'analog_bad'
        pretrain_net = Analog(config, device).to(device)
        model_name = '/analog_bad_model_Trans{:d}_snr-5.pth.tar'.format(
            config.channel_use,
        )
        pretrain_net.load_state_dict(torch.load(config.model_path + model_name))

        encoder_basic_dict = pretrain_net.encoder.state_dict()
        class_decoder_bad_dict = pretrain_net.class_decoder.state_dict()
        rec_decoder_bad_dict = pretrain_net.rec_decoder.state_dict()
        """"""

        config.net = 'scm'
        sp_model = SuperpositionNet(config, device).to(device)

        sp_model.basic_encoder.encoder.load_state_dict(encoder_basic_dict)
        sp_model.basic_decoder.rec_decoder.load_state_dict(rec_decoder_bad_dict)
        sp_model.basic_decoder.class_decoder.load_state_dict(class_decoder_bad_dict)

        sp_model.enhancement_encoder.encoder.load_state_dict(encoder_enhancement_dict)
        sp_model.enhancement_decoder.rec_decoder.load_state_dict(rec_decoder_good_dict)
        sp_model.enhancement_decoder.class_decoder.load_state_dict(class_decoder_good_dict)

        """Phase 1"""
        for param in sp_model.enhancement_encoder.parameters():
            param.requires_grad = False
        for param in sp_model.enhancement_decoder.parameters():
            param.requires_grad = False
        config.phase = '1'
        config.train_iters = 100
        config.lr = 2e-4
        print('Phase 1 training')
        train(config, sp_model, train_loader, test_loader, device)

        """Phase 2"""
        sp_model = SuperpositionNet(config, device).to(device)
        model_name = '/{}_SNR{:.1f}_{:.1f}_Trans{:d}_{}_phase{}_{}.pth.tar'.format(
            config.net,
            config.snr_train_good,
            config.snr_train_bad,
            config.channel_use,
            config.sp_mode,
            config.phase,
            config.a
        )
        sp_model.load_state_dict(torch.load(config.model_path + model_name))  # First loading the model of phase 1

        for param in sp_model.basic_encoder.parameters():
            param.requires_grad = False
        for param in sp_model.basic_decoder.parameters():
            param.requires_grad = False
        config.phase = '2'
        config.train_iters = 200
        config.lr = 2e-4
        print('Phase 2 training.')
        train(config, sp_model, train_loader, test_loader, device)

        """Phase 3"""
        model_name = '/{}_SNR{:.1f}_{:.1f}_Trans{:d}_{}_phase{}_{}.pth.tar'.format(
            config.net,
            config.snr_train_good,
            config.snr_train_bad,
            config.channel_use,
            config.sp_mode,
            config.phase,
            config.a
        )
        config.phase = '3'
        sp_model = SuperpositionNet(config, device).to(device)
        sp_model.load_state_dict(torch.load(config.model_path + model_name))
        for param in sp_model.parameters():
            param.requires_grad = True
        print('Phase 3 training.')
        config.train_iters = 100
        config.lr = 5e-5
        train(config, sp_model, train_loader, test_loader, device)

    elif config.net == 'analog_bad' or config.net == 'analog_good':
        if config.net == 'analog_bad':
            config.snr_train_good = -5
        elif config.net == 'analog_good':
            config.snr_train_good = 20
        config.train_iters = 150
        config.alpha_1 = 10

        net = Analog(config, device).to(device)
        train_analog(config, net, train_loader, test_loader, device)


def start_test(config, test_loader):
    """Testing phase-3 models"""
    config.phase = '3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SuperpositionNet(config, device).to(device)
    model_name = '/{}_SNR{:.1f}_{:.1f}_Trans{:d}_{}_phase3_{}_{}.pth.tar'.format(
        config.net,
        config.snr_train_good,
        config.snr_train_bad,
        config.channel_use,
        config.sp_mode,
        config.a,
        config.tradeoff
    )
    net.load_state_dict(torch.load(config.model_path + model_name))
    acc_total_bad, acc_total_good, psnr_total_bad, psnr_total_good, _, _ = EVAL_proposed(net, test_loader, device, config, -1)

    print('Bad receiver: acc: {:.4f}, psnr: {:.4f}; good receiver: acc: {:.4f}, psnr: {:.4f}'.format(
        acc_total_bad, psnr_total_bad, acc_total_good, psnr_total_good
    ))


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--channel_use', type=int, default=256)
    parser.add_argument('--net', type=str, default='scm')  # 'scm' , 'analog_good' or 'analog_bad' (the last two are for pretraining)
    parser.add_argument('--sp_mode', type=str, default='4and16')  # superposition mode, '4and16' or '4and4'
    parser.add_argument('--a', type=float, default=0.76)  # power allocation factor
    #  In the paper a = 0.76 when sp_mode = 4and16;
    #  a = 0.8 when sp_mode = 4and4
    parser.add_argument('--phase', type=str, default='1')
    parser.add_argument('--tradeoff', type=float, default=2)  # beta in eq. (21)
    parser.add_argument('--alpha_1', type=float, default=200)  # lambda_1 in eq. (19)
    parser.add_argument('--alpha_2', type=float, default=120)  # lambda_2 in eq. (20)
    parser.add_argument('--snr_train_good', type=float, default=20)
    parser.add_argument('--snr_train_bad', type=float, default=-5)
    parser.add_argument('--snr_test_good', type=float, default=20)
    parser.add_argument('--snr_test_bad', type=float, default=-5)
    # order of the pretrained model; is 64 for '4and16' superconstellation, and 16 for '4and4'
    parser.add_argument('--order', type=int, default=16)

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mode', type=str, default='train')  # 'train' or 'test'

    # misc
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--rec_path', type=str, default='./rec')
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--data_path', type=str, default='./trainingPlot')

    config = parser.parse_args()
    
    mischandler(config)
    main(config)
