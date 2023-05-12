import torch.nn as nn


def generate_conv_config(inc, rate=1, ks=(3, 3), st=(1, 1), dl=(1, 1), be=False):
    assert rate >= 1 and rate % 1 == 0 and ks[0] % 2 == 1 and ks[1] % 2 == 1
    return [inc, int(inc * rate), ks, st, ((ks[0]-1)//2 + dl[0] - 1, (ks[1]-1)//2 + dl[1] - 1), dl, inc, be]


def parse_conv_config(conv_config):
    inc, ouc, ks, st, pd, dl, gr, be = conv_config
    if not be:
        return nn.Conv2d(inc, ouc, ks, st, pd, dl, gr)
    else:
        return nn.Sequential(
            nn.Conv2d(inc, ouc, ks, st, pd, dl, gr),
            nn.BatchNorm2d(ouc),
            nn.ELU()
        )


def basic_conv_block(conv_configs):
    t = nn.Sequential()
    for conv_config in conv_configs:
        t.append(parse_conv_config(conv_config))
    return t


