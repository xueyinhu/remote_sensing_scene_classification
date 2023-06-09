import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torchsummary import summary
import numpy as np
from tqdm import tqdm

from config import get_config
from load_data import get_dataloader_2
from nets.try_net import FPN101

from torchvision.models import inception_v3


config = get_config()
train_dataloader, val_dataloader, test_dataloader = get_dataloader_2(config)
device = torch.device(config.device)
net = FPN101().to(device)
# net = inception_v3(num_classes=10, aux_logits=False).to(device)
summary(net, input_size=(3, 256, 256))
criterion = CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=config.lr)


def train():
    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(config.epoch_count):
        net.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        for idx, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == labels).cpu()
            nums += labels.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("epoch = {}, train acc = {:.3f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(train_epoch_loss)))
        with torch.no_grad():
            net.eval()
            val_epoch_loss = []
            acc, nums = 0, 0
            for idx, (inputs, labels) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_epoch_loss.append(loss.item())
                acc += sum(outputs.max(axis=1)[1] == labels).cpu()
                nums += labels.size()[0]
            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)
            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(val_epoch_loss)))
    torch.save(net, 'models/model.pth')


def test(model_path):
    net_t = torch.load(model_path)
    test_loss = []
    acc, nums = 0, 0
    for idx, (inputs, labels) in enumerate(tqdm(val_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = net_t(inputs)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())
        acc += sum(outputs.max(axis=1)[1] == labels).cpu()
        nums += labels.size()[0]
    print(np.average(test_loss))
    print(100 * acc / nums)


train()
test('./models/model.pth')

