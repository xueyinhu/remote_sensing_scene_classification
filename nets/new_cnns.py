import torch
import torch.nn as nn


class NewNet(nn.Module):
    def __init__(self, config, net_path='models/epoch_28_valid_loss_839899.375.pth'):
        super().__init__()
        self.config = config
        self.head = torch.load(net_path).head
        self.body = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), (2, 2), padding=(1, 1), groups=256),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, (1, 1)),
            nn.Conv2d(512, 512+256, (7, 1), (3, 1), padding=(3, 0), groups=256),
            nn.Conv2d(512+256, 1024, (1, 7), (1, 3), padding=(0, 3), groups=256),
            nn.BatchNorm2d(1024),
            nn.ELU(),
            nn.Conv2d(1024, 1024, (1, 1)),
        )
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.Dropout(.2),
            nn.Linear(128, config.class_num)
        )

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std * 0.5)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x0):
        x1 = self.head[0](x0)
        x2 = self.head[1](x1)
        x3 = self.head[2](x2)
        mu = self.head[3](x3)
        log_std = self.head[4](x3)
        z = self.reparametrize(mu, log_std)
        return self.tail(self.body(z))

