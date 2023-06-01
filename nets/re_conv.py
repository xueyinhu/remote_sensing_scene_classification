import torch
import torch.nn as nn


def conv_block(inc, ouc):
    return nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            nn.Conv2d(inc, inc, (3, 1), (2, 1), padding=(1, 0), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.Conv2d(inc, inc, (1, 3), (1, 2), padding=(0, 1), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc),
            nn.ELU(),
        ),
        nn.Sequential(
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (3, 1), (2, 1), padding=(1, 0), groups=1),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            # # nn.Conv2d(inc, inc, (1, 1)),
            # # nn.BatchNorm2d(inc),
            # # nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 3), (1, 2), padding=(0, 1), groups=1),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            # # nn.Conv2d(inc, inc, (1, 1)),
            # # nn.BatchNorm2d(inc),
            # # nn.ELU(),
            nn.Conv2d(inc, inc, (3, 1), (2, 1), padding=(1, 0), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.Conv2d(inc, inc, (1, 3), (1, 2), padding=(0, 1), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.ConvTranspose2d(inc, inc, (3, 1), (2, 1), padding=(1, 0), output_padding=(1, 0), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.ConvTranspose2d(inc, inc, (1, 3), (1, 2), padding=(0, 1), output_padding=(0, 1), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        ),
        nn.Sequential(
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            nn.Conv2d(inc, inc, (3, 1), (2, 1), padding=(1, 0), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.Conv2d(inc, inc, (1, 3), (1, 2), padding=(0, 1), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.Conv2d(inc, inc, (3, 1), (1, 1), padding=(1, 0), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.Conv2d(inc, inc, (1, 3), (1, 1), padding=(0, 1), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc),
            nn.ELU(),
        ),
        nn.Sequential(
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            nn.Conv2d(inc, inc, (3, 1), (2, 1), padding=(1, 0), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.Conv2d(inc, inc, (1, 3), (1, 2), padding=(0, 1), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            # nn.Conv2d(inc, inc, (3, 1), (1, 1), padding=(2, 0), dilation=(2, 1), groups=1),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            # # nn.Conv2d(inc, inc, (1, 1)),
            # # nn.BatchNorm2d(inc),
            # # nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 3), (1, 1), padding=(0, 2), dilation=(1, 2), groups=1),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            # # nn.Conv2d(inc, inc, (1, 1)),
            # # nn.BatchNorm2d(inc),
            # # nn.ELU(),
            nn.Conv2d(inc, inc, (3, 1), (1, 1), padding=(2, 0), dilation=(2, 1), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            # nn.Conv2d(inc, inc, (1, 1)),
            # nn.BatchNorm2d(inc),
            # nn.ELU(),
            nn.Conv2d(inc, inc, (1, 3), (1, 1), padding=(0, 2), dilation=(1, 2), groups=1),
            nn.BatchNorm2d(inc),
            nn.ELU(),
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc),
            nn.ELU(),
        ),
        nn.Conv2d(inc * 4, ouc, (1, 1))
    ])


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 32, (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.body = nn.ModuleList([
            # conv_block(8),
            conv_block(32, 128),
            conv_block(128, 256),
            conv_block(256, 256),
            conv_block(256, 256),
            conv_block(256, 256),
            conv_block(256, 256),
            # conv_block(512, 512),
            # conv_block(2048)
        ])
        self.tail = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(.4),
            nn.Linear(1024, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.head(x)
        for m in self.body:
            # print(m[0](x).shape)
            # print(m[1](x).shape)
            # print(m[2](x).shape)
            # print(m[3](x).shape)
            x = m[4](torch.cat([m[0](x), m[1](x), m[2](x), m[3](x)], dim=1))
        return self.tail(x)
