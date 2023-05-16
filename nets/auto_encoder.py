import site

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.ModuleList([
            nn.Conv2d(3, 8, (5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(8), nn.ELU(),
            nn.Conv2d(8, 16, (7, 1), (5, 1), padding=(6, 0), dilation=(2, 1), groups=8),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.Conv2d(16, 32, (1, 7), (1, 5), padding=(0, 6), dilation=(1, 2), groups=16),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 64, (1, 1)),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 64, (9, 1), (5, 1), padding=(4, 0), groups=64),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 64, (1, 9), (1, 5), padding=(0, 4), groups=64),
            nn.BatchNorm2d(64), nn.ELU(),
            # output (12 * 12 * 64)
        ])
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(.2),
        )
        self.tail = nn.ModuleList([
            nn.ConvTranspose2d(64, 64, (1, 9), (1, 5), padding=(0, 4), groups=64),
            nn.ConvTranspose2d(64, 64, (9, 1), (5, 1), padding=(4, 0), groups=64),
            nn.ConvTranspose2d(64, 32, (1, 1)),
            nn.ConvTranspose2d(32, 16, (1, 7), (1, 5), padding=(0, 6), dilation=(1, 2), groups=16),
            nn.ConvTranspose2d(16, 8, (7, 1), (5, 1), padding=(6, 0), dilation=(2, 1), groups=8),
            nn.ConvTranspose2d(8, 3, (5, 5), stride=(2, 2), padding=(2, 2)),
        ])

    def forward(self, x):
        sl = []
        i = 0
        for m in self.head:
            if i % 3 == 0:
                sl.append(x.shape)
            x = m(x)
            i += 1
        s = x.shape
        y = torch.reshape(self.body(x), s)
        sl.reverse()
        i = 0
        for m in self.tail:
            y = m(y, output_size=sl[i])
            i += 1
        return y



