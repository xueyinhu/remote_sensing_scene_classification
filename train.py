import torch
from torchsummary import summary

from config import get_config
from load_data import get_dataloader
from nets.try_cnns import TryCNNs

config = get_config()
train_dataloader, val_dataloader, _ = get_dataloader(config)
net = TryCNNs(config)
summary(net, input_size=(3, 256, 256))



