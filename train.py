import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt

from config import get_config
from load_data import get_dataloader
from nets.try_cnns import TryCNNs

config = get_config()
train_dataloader, val_dataloader, _ = get_dataloader(config)
device = torch.device(config.device)
net = TryCNNs(config).to(device)
summary(net, input_size=(3, 256, 256))
criterion = CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=config.lr)


def train():
    running_loss = 0.
    for _, (image, label) in enumerate(train_dataloader):
        image, label = image.to(device), label.to(device)
        out_p = net(image)
        loss = criterion(out_p, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def val():
    correct, total = 0, 0
    with torch.no_grad():
        for _, (image, label) in enumerate(val_dataloader):
            image, label = image.to(device), label.to(device)
            out_p = net(image)
            prediction = out_p.argmax(dim=1)
            total += label.size(0)
            correct += sum(prediction == label)
        print('Accuracy on val dataset: ( %d / %d ) -> %d %%' % (correct, total, 100 * correct / total))


if __name__ == '__main__':
    loss_list = []
    best_loss = 1e3
    for epoch in range(config.epoch_count):
        loss_list.append(train())
        print("Epoch: %d, loss: %5f" % (epoch + 1, loss_list[epoch]))
        val()
        if loss_list[epoch] < best_loss:
            best_loss = loss_list[epoch]
            torch.save(net, "models/best_loss_%f.pth" % best_loss)
    plt.title('Graph')
    plt.plot(range(config.epoch_count), loss_list)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
