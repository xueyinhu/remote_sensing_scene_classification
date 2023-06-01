import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBNBlock(inc, ouc, ks, st, ac=True, dl=1):
    cb = nn.Sequential(
        nn.Conv2d(
            inc, ouc, ks, st,
            padding=((ks-1) // 2) + dl - 1,
            groups=1 if ks == 1 else inc if inc <= ouc else ouc,
            dilation=dl,
            bias=False
        ),
        nn.BatchNorm2d(ouc)
    )
    if ac:
        cb.append(nn.ELU(inplace=True))
    return cb


class CBlockA(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
            ConvBNBlock(inc, inc // 2, 1, 1, False),
        )
        self.b2 = nn.Sequential(
            ConvBNBlock(inc, inc // 2, 1, 1, False),
            ConvBNBlock(inc // 2, inc // 2, 3, 2),
        )
        self.b3 = nn.Sequential(
            ConvBNBlock(inc, inc // 2, 1, 1, False),
            ConvBNBlock(inc // 2, inc // 2, 3, 1),
            ConvBNBlock(inc // 2, inc // 2, 3, 2),
        )
        self.b4 = nn.Sequential(
            ConvBNBlock(inc, inc // 2, 1, 1, False),
            ConvBNBlock(inc // 2, inc // 2, 3, 1),
            ConvBNBlock(inc // 2, inc // 2, 3, 2, dl=2),
        )
        self.t = ConvBNBlock(inc * 2, inc, 1, 1, False)

    def forward(self, x):
        return self.t(torch.cat([
            self.b1(x), self.b2(x), self.b3(x), self.b4(x)
        ], dim=1))
    

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.cm = CBAM(planes * 4)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.down_sample is not None:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        out = self.cm(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, layers):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.top_layer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.lat_layer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.s2 = nn.Sequential(
            CBlockA(256),
        )
        self.s3 = nn.Sequential(
            CBlockA(512),
        )
        self.s4 = nn.Sequential(
            CBlockA(768),
        )
        self.s5 = nn.Sequential(
            CBlockA(1024),
        )
        self.tail = nn.Sequential(
            ConvBNBlock(1024, 1024, 3, 2),
            ConvBNBlock(1024, 1024, 1, 1, False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, down_sample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.top_layer(c5)
        p4 = self._upsample_add(p5, self.lat_layer1(c4))
        p3 = self._upsample_add(p4, self.lat_layer2(c3))
        p2 = self._upsample_add(p3, self.lat_layer3(c2))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        p2 = self.s2(p2)
        p3 = self.s3(torch.cat([p2, p3], 1))
        p4 = self.s4(torch.cat([p3, p4], 1))
        p5 = self.s5(torch.cat([p4, p5], 1))
        y = self.tail(p5)

        return y


def FPN101():
    return FPN(Bottleneck, [3, 4, 6, 3])





