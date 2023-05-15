import torch
from torchsummary import summary

from config import get_config
from nets.try_cnns import TryCNNs

config = get_config()
device = torch.device(config.device)
net = TryCNNs(config).to(device)
summary(net, input_size=(3, 600, 600))
