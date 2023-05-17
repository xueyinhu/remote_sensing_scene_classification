import torch
import torch.nn as nn


class NewNet(nn.Module):
    def __init__(self, config, net_path='models/epoch_26_valid_loss_0.005431759171187878.pth'):
        super().__init__()
        self.config = config
        self.head = torch.load(net_path).head
        self.body = nn.Sequential(
        )
        self.tail = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(.2),
            nn.Linear(12 * 12 * 64, config.class_num)
        )

    def forward(self, x):
        for h in self.head:
            x = h(x)
        return self.tail(x)

