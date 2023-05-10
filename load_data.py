import torch
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import get_image_list, get_image_for_path


def get_dataloader(config):
    modes = ['train', 'val', 'test']
    datasets = [MyDataset(config.txt_path, config.split_rate, mode) for mode in modes]
    return [
        DataLoader(
            datasets[modes.index(mode)],
            batch_size=config.batch_size,
            shuffle=(mode == 'train')
        ) for mode in modes
    ]


class MyDataset(Dataset):
    def __init__(self, path, split_rate, mode):
        self.lines = get_image_list(path, split_rate[(['train', 'val', 'test']).index(mode)])
        self.trans = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = get_image_for_path(self.lines[index][0])
        return self.trans(image), torch.tensor(int(self.lines[index][1]), dtype=torch.int64)

    def __len__(self):
        return len(self.lines)
