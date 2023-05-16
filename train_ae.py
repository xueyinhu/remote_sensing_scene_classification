import torch
import numpy as np
from torch.nn import MSELoss
import torch.optim as optim
from torchsummary import summary

from config import get_config
from load_data import get_dataloader_2
from nets.auto_encoder import AutoEncoder

config = get_config()
train_dataloader, valid_dataloader, test_dataloader = get_dataloader_2(config)
device = torch.device(config.device)
net = AutoEncoder().to(device)
summary(net, input_size=(3, 600, 600))
loss_func = MSELoss()
optimizer = optim.Adam(net.parameters(), lr=config.lr)


def train():
    for epoch in range(config.epoch_count):
        print(epoch + 1)
        train_loss, valid_loss = [], []
        for _, (x, _) in enumerate(train_dataloader):
            train_loss = []
            x = x.to(device)
            y = net(x)
            loss = loss_func(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print(np.average(train_loss))
        for _, (x, _) in enumerate(valid_dataloader):
            valid_loss = []
            x = x.to(device)
            with torch.no_grad():
                y = net(x)
            loss = loss_func(x, y)
            valid_loss.append(loss.item())
        print(np.average(valid_loss))
        torch.save(net, 'models/epoch_'+str(epoch + 1)+'_valid_loss_'+str(np.average(valid_loss))+'.pth')


def test(model_path):
    net_t = torch.load(model_path)
    test_loss = []
    for _, (x, _) in enumerate(test_dataloader):
        x = x.to(device)
        with torch.no_grad():
            y = net_t(x)
        loss = loss_func(x, y)
        test_loss.append(loss.item())
    print(np.average(test_loss))


if __name__ == '__main__':
    # train()
    test('models/epoch_26_valid_loss_0.005431759171187878.pth')

