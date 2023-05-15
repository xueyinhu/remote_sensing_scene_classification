import torch
from torch.nn import MSELoss
import torch.optim as optim
from torchsummary import summary

from config import get_config
from load_data import get_dataloader_2
from nets.auto_encoder import AutoEncoder

config = get_config()
train_dataloader, val_dataloader, test_dataloader = get_dataloader_2(config)
device = torch.device(config.device)
net = AutoEncoder().to(device)
summary(net, input_size=(3, 600, 600))
criterion = MSELoss()
optimizer = optim.Adam(net.parameters(), lr=config.lr)

