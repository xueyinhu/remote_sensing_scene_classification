import torch
import numpy as np
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms

from config import get_config
from load_data import get_dataloader_2
from nets.vae import VAE
from utils import get_image_for_path_pil

config = get_config()
train_dataloader, valid_dataloader, test_dataloader = get_dataloader_2(config)
device = torch.device(config.device)
net = VAE().to(device)
summary(net, input_size=(3, 600, 600))
optimizer = optim.Adam(net.parameters(), lr=config.lr)


def train():
    for epoch in range(config.epoch_count):
        print(epoch + 1)
        train_loss, valid_loss = [], []
        for _, (x, _) in enumerate(train_dataloader):
            train_loss = []
            x = x.to(device)
            y, mu, log_std = net(x)
            loss = net.loss_function(x, y, mu, log_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print(np.average(train_loss))
        for _, (x, _) in enumerate(valid_dataloader):
            valid_loss = []
            x = x.to(device)
            with torch.no_grad():
                y, mu, log_std = net(x)
            loss = net.loss_function(x, y, mu, log_std)
            valid_loss.append(loss.item())
        print(np.average(valid_loss))
        torch.save(net, 'models/epoch_'+str(epoch + 1)+'_valid_loss_'+str(np.average(valid_loss))+'.pth')


def test(model_path):
    net_t = torch.load(model_path)
    test_loss = []
    for _, (x, _) in enumerate(test_dataloader):
        x = x.to(device)
        with torch.no_grad():
            y, mu, log_std = net_t(x)
        loss = net.loss_function(x, y, mu, log_std)
        test_loss.append(loss.item())
    print(np.average(test_loss))


def show_image(model_path, image_path):
    net_t = torch.load(model_path)
    image = get_image_for_path_pil(image_path)
    x = transforms.ToTensor()(image)
    x = torch.unsqueeze(x, dim=0).to(device)
    y, _, _ = net_t(x)
    result = torch.squeeze(y, dim=0).cpu()
    transforms.ToPILImage()(result).show()


if __name__ == '__main__':
    train()
    # test('models/epoch_2_valid_loss_5152471.5.pth')
    # show_image('models/epoch_29_valid_loss_132667.984375.pth', 'airport_2.jpg')

