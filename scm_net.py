from torch import nn
import torch
from torch.nn import init
from basic_module import awgn, DepthToSpace, Decoder_Recon, Decoder_Class, normalize, Encoder
from torch.nn.functional import gumbel_softmax


def modulation(probs, order, device, p=1):
    order_sqrt = int(order ** 0.5)
    prob_z = gumbel_softmax(probs, hard=False)
    discrete_code = gumbel_softmax(probs, hard=True, tau=1.7)

    if order_sqrt == 2:
        const = [1, -1]
    elif order_sqrt == 4:
        const = [-3, -1, 1, 3]
    elif order_sqrt == 8:
        const = [-7, -5, -3, -1, 1, 3, 5, 7]

    const = torch.tensor(const, dtype=torch.float).to(device)

    ave_p = torch.mean(const ** 2)
    const = const / ave_p ** 0.5

    temp = discrete_code * const
    output = torch.sum(temp, dim=2)

    return output, prob_z


class BasicEncoder(nn.Module):
    def __init__(self, config, device, order):
        super(BasicEncoder, self).__init__()

        self.device = device
        self.config = config
        self.order = order

        if self.order == 4:
            self.num_category = 2  # for I channel and Q channel, the number of probability categories is sqrt(order)
        elif self.order == 16:
            self.num_category = 4

        self.encoder = Encoder(config)
        self.ave_pool = nn.AvgPool2d(2)
        self.prob_convs = nn.Sequential(
            nn.Linear(config.channel_use * 2, config.channel_use * 2 * self.num_category),
            nn.PReLU()
        )

    def reparameterize(self, probs):
        code, probs = modulation(probs, self.order, self.device)
        return code, probs

    def forward(self, x):
        u = self.encoder(x)
        u = self.ave_pool(u).reshape(x.shape[0], -1)
        z_u = self.prob_convs(u).reshape(x.shape[0], -1, self.num_category)
        z_u, probs = self.reparameterize(z_u)
        return u, z_u, probs


class EnhancementEncoder(nn.Module):
    def __init__(self, config, device, order):
        super(EnhancementEncoder, self).__init__()

        self.config = config
        self.device = device
        self.order = order

        if self.order == 4:
            self.num_category = 2
        elif self.order == 16:
            self.num_category = 4

        self.encoder = Encoder(config)
        self.ave_pool = nn.AvgPool2d(2)
        self.prob_convs = nn.Sequential(
            nn.Linear(config.channel_use * 2, config.channel_use * 2 * self.num_category),
            nn.PReLU()
        )

        self.linear_transform = nn.Linear(in_features=config.channel_use * 2, out_features=config.channel_use * 2)  # demodulator

    def reparameterize(self, probs):
        code, probs = modulation(probs, self.order, self.device)
        return code, probs

    def forward(self, x, u):
        v = self.encoder(x)
        v = self.ave_pool(v).reshape(x.shape[0], -1)

        if self.config.phase == '1':
            u_residual = v  # do not activate demodulator in phase 1
            z_v = self.prob_convs(v).reshape(x.shape[0], -1, self.num_category)
        else:
            u_residual = v - self.linear_transform(u)
            z_v = self.prob_convs(u_residual).reshape(x.shape[0], -1, self.num_category)

        z_v, probs = self.reparameterize(z_v)
        return u_residual, z_v, probs


class BasicDecoder(nn.Module):
    def __init__(self, config):
        super(BasicDecoder, self).__init__()
        self.config = config

        self.rec_decoder = Decoder_Recon(config)

        self.layer_width = int(config.channel_use * 2 / 8)
        self.Half_width = int(config.channel_use * 2 / 2)
        self.class_decoder = Decoder_Class(self.Half_width, self.layer_width, 20)

    def forward(self, z_hat):
        rec = self.rec_decoder(z_hat.reshape(z_hat.shape[0], -1, 4, 4))
        pred = self.class_decoder(z_hat)
        return rec, pred


class EnhancementDecoder(nn.Module):
    def __init__(self, config):
        super(EnhancementDecoder, self).__init__()

        self.rec_decoder = Decoder_Recon(config)

        self.layer_width = int(config.channel_use * 2 / 8)
        self.Half_width = int(config.channel_use * 2 / 2)
        self.class_decoder = Decoder_Class(self.Half_width, self.layer_width, 100)

    def forward(self, z_hat):
        rec = self.rec_decoder(z_hat.reshape(z_hat.shape[0], -1, 4, 4))
        pred = self.class_decoder(z_hat)
        return rec, pred


class SuperpositionNet(nn.Module):
    def __init__(self, config, device):
        super(SuperpositionNet, self).__init__()
        self.config = config
        self.device = device

        if self.config.sp_mode == '4and4':
            low_order = 4
            high_order = 4
            self.a = self.config.a
        elif self.config.sp_mode == '4and16':
            low_order = 4
            high_order = 16
            self.a = self.config.a

        self.basic_encoder = BasicEncoder(self.config, self.device, low_order)
        self.enhancement_encoder = EnhancementEncoder(self.config, self.device, high_order)

        self.basic_decoder = BasicDecoder(self.config)
        self.enhancement_decoder = EnhancementDecoder(self.config)

        self.initialize_weights()

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):

        u, z_u, probs_u = self.basic_encoder(x)

        u_residual, z_v, probs_v = self.enhancement_encoder(x, u)

        _, z_u_norm = normalize(z_u)
        _, z_v_norm = normalize(z_v)

        z = (self.a ** 0.5) * z_u_norm + ((1 - self.a) ** 0.5) * z_v_norm
        _, z = normalize(z)

        # into channel
        if self.config.mode == 'train':
            z_hat_bad = awgn(self.config.snr_train_bad, z, self.device)
            if self.config.snr_train_good == 50:
                z_hat_good = z
            else:
                z_hat_good = awgn(self.config.snr_train_good, z, self.device)
        elif self.config.mode == 'test':
            z_hat_bad = awgn(self.config.snr_test_bad, z, self.device)
            if self.config.snr_test_good == 50:
                z_hat_good = z
            else:
                z_hat_good = awgn(self.config.snr_test_good, z, self.device)

        # receiver 1
        rec_bad, pred_bad = self.basic_decoder(z_hat_bad)

        #  receiver 2
        rec_good, pred_good = self.enhancement_decoder(z_hat_good)

        return u_residual, z, z_hat_bad, z_hat_good, rec_bad, pred_bad, rec_good, pred_good

