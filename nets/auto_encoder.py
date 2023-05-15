import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 8, (5, 5), stride=(2, 2), padding=(2, 2)),
            nn.Conv2d(8, 16, (7, 1), (5, 1), padding=(6, 0), dilation=(2, 1), groups=8),
            nn.Conv2d(16, 32, (1, 7), (1, 5), padding=(0, 6), dilation=(1, 2), groups=16),
            nn.Conv2d(32, 64, (1, 1)),
            nn.Conv2d(64, 64, (9, 1), (5, 1), padding=(4, 0), groups=64),
            nn.Conv2d(64, 64, (1, 9), (1, 5), padding=(0, 4), groups=64),
            # output (12 * 12 * 64)
        )
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(.2),
        )
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (1, 9), (1, 5), padding=(0, 4), groups=64),
            nn.ConvTranspose2d(64, 64, (9, 1), (5, 1), padding=(4, 0), groups=64),
            nn.ConvTranspose2d(64, 32, (1, 1)),
            nn.ConvTranspose2d(32, 16, (1, 7), (1, 5), padding=(0, 6), dilation=(1, 2), groups=16),
            nn.ConvTranspose2d(16, 8, (7, 1), (5, 1), padding=(6, 0), dilation=(2, 1), groups=8),
            nn.ConvTranspose2d(8, 3, (5, 5), stride=(2, 2), padding=(2, 2)),
        )

    def forward(self, x):
        y = self.head(x)
        s = y.shape
        y = self.body(y)
        y = torch.reshape(y, s)
        y = self.tail(y)
        return y



