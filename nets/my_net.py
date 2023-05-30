import torch
import torch.nn as nn


def conv_bn_block(inc, ouc, ks, st, pd, dl, gr):
    return nn.Sequential(
        nn.Conv2d(inc, ouc, ks, stride=st, padding=pd, dilation=dl, groups=gr),
        nn.BatchNorm2d(ouc),
        nn.ELU()
    )


def conv_t_bn_block(inc, ouc, ks, st, pd, dl, gr):
    return nn.Sequential(
        nn.Conv2d(inc, ouc, ks, st, pd, dilation=dl, groups=gr),
        nn.BatchNorm2d(ouc),
        nn.ELU()
    )


def branch_1(inc):
    return nn.Sequential(
        conv_bn_block(inc, inc, (3, 1), (1, 1), (1, 0), (1, 1), inc),
        conv_t_bn_block(inc, inc, (3, 1), (1, 1), (1, 0), (1, 1), inc),
        conv_bn_block(inc, inc, (1, 3), (1, 1), (0, 1), (1, 1), inc),
        conv_t_bn_block(inc, inc, (1, 3), (1, 1), (0, 1), (1, 1), inc),
    )


def branch_2(inc):
    return nn.Sequential(
        conv_bn_block(inc, inc, (3, 1), (1, 1), (2, 0), (2, 1), inc),
        conv_bn_block(inc, inc, (1, 3), (1, 1), (0, 2), (1, 2), inc),
        conv_bn_block(inc, inc, (3, 1), (1, 1), (2, 0), (2, 1), inc),
        conv_bn_block(inc, inc, (1, 3), (1, 1), (0, 2), (1, 2), inc),
    )


class Block_1(nn.Module):
    def __init__(self, inc, flag):
        super().__init__()
        self.inc = inc
        self.flag = flag
        self.b1 = branch_1(inc // 2)
        self.b2 = branch_2(inc // 2)
        self.cb = nn.Sequential(
            nn.Conv2d(inc, inc, (1, 1)),
            nn.BatchNorm2d(inc)
        )

    def forward(self, x):
        x = self.cb(x)
        if self.flag:
            x1, x2 = torch.split(x, dim=1, split_size_or_sections=self.inc // 2)
        else:
            x2, x1 = torch.split(x, dim=1, split_size_or_sections=self.inc // 2)
        x1 = self.b1(x1)
        x2 = self.b2(x2)
        y = torch.cat([x1, x2], dim=1)
        return y


class Block_2(nn.Module):
    def __init__(self, inc, rate=1):
        super().__init__()
        self.ouc = int(inc * rate)
        self.b1 = nn.Sequential(
            conv_bn_block(inc, inc, (1, 1), (1, 1), (0, 0), (1, 1), 1),
            conv_bn_block(inc, inc, (3, 1), (2, 1), (1, 0), (1, 1), inc),
            conv_bn_block(inc, inc, (1, 3), (1, 2), (0, 1), (1, 1), inc),
        )
        self.b2 = nn.Sequential(
            conv_bn_block(inc, inc, (1, 1), (1, 1), (0, 0), (1, 1), 1),
            conv_bn_block(inc, inc, (3, 1), (1, 1), (1, 0), (1, 1), inc),
            conv_bn_block(inc, inc, (1, 3), (1, 1), (0, 1), (1, 1), inc),
            conv_bn_block(inc, inc, (3, 1), (2, 1), (1, 0), (1, 1), inc),
            conv_bn_block(inc, inc, (1, 3), (1, 2), (0, 1), (1, 1), inc),
        )

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        y = torch.cat([x1, x2], dim=1)
        return y


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        self.body_1 = nn.Sequential(
            Block_2(16),
            Block_1(32, True),
            Block_1(32, False),
            Block_2(32),
            Block_1(64, True),
            Block_1(64, False),
            Block_2(64),
            Block_1(128, True),
            Block_1(128, False),
            Block_2(128),
        )
        self.body_2 = nn.Sequential(
            Block_1(256, True),
            Block_1(256, False),
            Block_2(256),
        )
        self.body_3 = nn.Sequential(
            Block_1(512, True),
            Block_1(512, False),
            Block_2(512),
        )
        self.body_4 = nn.Sequential(
            Block_2(256),
            Block_2(512),
        )
        self.body_5 = nn.Sequential(
            Block_2(512),
        )
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(.4),
            nn.Linear(1024 * 3, 45),
        )

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body_1(x0)
        x2 = self.body_2(x1)
        x3 = self.body_3(x2)
        x4 = self.body_4(x1)
        x5 = self.body_5(x2)
        x = torch.cat([x4, x5, x3], dim=1)
        return self.tail(x)










