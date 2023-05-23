import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from utils import get_image_list_for_txt, get_image_for_path, get_image_list_for_path, get_image_for_path_pil


def get_dataloader_1(config):
    modes = ['train', 'val', 'test']
    datasets = [MyDataset_1(config.txt_path, config.split_rate, mode) for mode in modes]
    return [
        DataLoader(
            datasets[modes.index(mode)],
            batch_size=config.batch_size,
            shuffle=(mode == 'train')
        ) for mode in modes
    ]


def get_dataloader_2(config):
    datasets = MyDataset_2(config.data_path)
    split_length = [int(datasets.__len__() * (i[1] * 10 - i[0] * 10) / 10) for i in config.split_rate]
    datasets = random_split(datasets, lengths=split_length)
    return [
        DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        ) for dataset in datasets
    ]


class MyDataset_1(Dataset):
    def __init__(self, path, split_rate, mode):
        self.lines = get_image_list_for_txt(path, split_rate[(['train', 'val', 'test']).index(mode)])
        self.trans = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = get_image_for_path(self.lines[index][0])
        return self.trans(image), torch.tensor(int(self.lines[index][1]), dtype=torch.int64)

    def __len__(self):
        return len(self.lines)


class MyDataset_2(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_list = get_image_list_for_path(path)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((300, 300))
        ])

    def __getitem__(self, index):
        data = self.data_list[index].split(' ')
        return self.trans(get_image_for_path_pil(data[0])), torch.tensor(int(data[1]))

    def __len__(self):
        return len(self.data_list)




