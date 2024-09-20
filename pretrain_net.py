from torch import nn
from torch.nn import init
from basic_module import awgn, Decoder_Recon, Decoder_Class, normalize, Encoder


class Analog(nn.Module):
    def __init__(self, config, device):
        super(Analog, self).__init__()

        self.config = config
        self.device = device
        self.layer_width = int(config.channel_use * 2 / 8)
        self.Half_width = int(config.channel_use * 2 / 2)

        self.encoder = Encoder(config)
        self.conv_last = nn.Conv2d(config.channel_use * 2, config.channel_use * 2, 2, 1, 0)

        self.rec_decoder = Decoder_Recon(config)
        if config.net == 'analog_bad':
            self.class_decoder = Decoder_Class(self.Half_width, self.layer_width, 20)
        elif config.net == 'analog_good':
            self.class_decoder = Decoder_Class(self.Half_width, self.layer_width, 100)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.encoder(x)
        z = self.conv_last(x).reshape(x.shape[0], -1)

        _, z_norm = normalize(z)

        if self.config.mode == 'train':
            z_hat = awgn(self.config.snr_train_good, z_norm, self.device)
        elif self.config.mode == 'test':
            z_hat = awgn(self.config.snr_test_good, z_norm, self.device)

        rec = self.rec_decoder(z_hat.reshape(x.shape[0], -1, 4, 4))
        pred = self.class_decoder(z_hat)

        return rec, pred