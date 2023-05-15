import torch
import torch.nn as nn


class TryCNNs(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 8, (23, 23), padding=(11, 11)),
            nn.Conv2d(8, 8, (1, 1)),
            basic_conv_block(8, 8, (17, 17), (1, 1), (8, 8), (1, 1), 8),
        )
        self.cbs = nn.Sequential()
        [self.cbs.append(ConvBlock(8 * 2 ** i)) for i in range(config.conv_block_count)]
        self.body = nn.Sequential(
            basic_conv_block(64, 128, (11, 11), (5, 5), (5, 5), (1, 1), 64),
            basic_conv_block(128, 256, (11, 11), (5, 5), (5, 5), (1, 1), 128)
        )
        self.tail = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(.2),
            nn.Linear(256 * 9, 64),
            nn.Linear(64, config.class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.tail(self.body(self.cbs(self.head(x))))


def basic_conv_block(inc, ouc, ks, st, pd, dl, gr):
    return nn.Sequential(
        nn.Conv2d(inc, ouc, kernel_size=ks, stride=st, padding=pd, dilation=dl, groups=gr),
        nn.BatchNorm2d(ouc),
        nn.ELU()
    )


class ConvBlock(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.ms = nn.ModuleList([
            nn.Sequential(
                basic_conv_block(inc, inc // 2, (1, 1), (1, 1), (0, 0), (1, 1), 1),
                basic_conv_block(inc // 2, inc // 2, (13, 13), (1, 1), (6, 6), (1, 1), inc // 2),
                basic_conv_block(inc // 2, inc, (13, 13), (2, 2), (6, 6), (1, 1), inc // 2),
            ),
            nn.Sequential(
                basic_conv_block(inc, inc // 2, (1, 1), (1, 1), (0, 0), (1, 1), 1),
                basic_conv_block(inc // 2, inc // 2, (13, 13), (1, 1), (6, 6), (1, 1), inc // 2),
                basic_conv_block(inc // 2, inc, (13, 13), (2, 2), (12, 12), (2, 2), inc // 2),
            ),
            nn.Sequential(
                basic_conv_block(inc, inc // 2, (1, 1), (1, 1), (0, 0), (1, 1), 1),
                basic_conv_block(inc // 2, inc // 2, (13, 13), (1, 1), (12, 12), (2, 2), inc // 2),
                basic_conv_block(inc // 2, inc, (13, 13), (2, 2), (6, 6), (1, 1), inc // 2),
            ),
            nn.Sequential(
                basic_conv_block(inc, inc // 2, (1, 1), (1, 1), (0, 0), (1, 1), 1),
                basic_conv_block(inc // 2, inc // 2, (13, 13), (1, 1), (12, 12), (2, 2), inc // 2),
                basic_conv_block(inc // 2, inc, (13, 13), (2, 2), (12, 12), (2, 2), inc // 2),
            )
        ])
        self.rs = nn.Sequential(
            basic_conv_block(inc * 4, inc * 2, (1, 1), (1, 1), (0, 0), (1, 1), 1),
            basic_conv_block(inc * 2, inc * 2, (9, 9), (1, 1), (4, 4), (1, 1), inc * 2)
        )

    def forward(self, x):
        y = torch.cat([m(x) for m in self.ms], dim=1)
        return self.rs(y)
