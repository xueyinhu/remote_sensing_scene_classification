import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 8, (3, 3), (1, 1), padding=(1, 1)),
                nn.Conv2d(8, 16, (3, 3), (1, 1), padding=(1, 1)),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, (3, 3), (2, 2), padding=(1, 1), groups=16),
                nn.ELU(),
                nn.Conv2d(32, 32, (1, 1)),
                nn.ELU(),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, (3, 3), (2, 2), padding=(1, 1), groups=32),
                nn.ELU(),
                nn.Conv2d(64, 64, (1, 1)),
                nn.ELU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, (1, 7), (1, 5), padding=(0, 3), groups=64),
                nn.ELU(),
                nn.Conv2d(128, 256, (7, 1), (5, 1), padding=(3, 0), groups=128),
                nn.ELU(),
                nn.Conv2d(256, 256, (1, 1)),
                nn.ELU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, (1, 7), (1, 5), padding=(0, 3), groups=64),
                nn.ELU(),
                nn.Conv2d(128, 256, (7, 1), (5, 1), padding=(3, 0), groups=128),
                nn.ELU(),
                nn.Conv2d(256, 256, (1, 1)),
                nn.ELU(),
            )
        ])
        self.body = self.reparametrize
        self.tail = nn.ModuleList([
            nn.ConvTranspose2d(256, 64, (7, 7), (5, 5), padding=(3, 3), groups=64),
            nn.Sequential(
                nn.ELU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, (1, 1)),
                nn.ELU(),
            ),
            nn.ConvTranspose2d(64, 32, (3, 3), (2, 2), padding=(1, 1), groups=32),
            nn.Sequential(
                nn.ELU(),
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, (1, 1)),
                nn.ELU(),
            ),
            nn.ConvTranspose2d(32, 16, (3, 3), (2, 2), padding=(1, 1), groups=16),
            nn.Conv2d(16, 3, (3, 3), (1, 1), padding=(1, 1)),
        ])

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
        z = self.body(mu, log_std)
        y = self.tail[0](z, output_size=x3.size())
        y = self.tail[2](self.tail[1](y))
        y = self.tail[3](y, output_size=x2.size())
        y = self.tail[5](self.tail[4](y))
        y = self.tail[6](y, output_size=x1.size())
        y = self.tail[7](y)
        return y, mu, log_std

    def loss_function(self, x, recon, mu, log_std) -> torch.Tensor:
        recon_loss = F.l1_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * (1 + log_std - mu.pow(2) - torch.exp(log_std))
        kl_loss = torch.sum(kl_loss)
        loss = 2 * recon_loss + kl_loss
        return loss
