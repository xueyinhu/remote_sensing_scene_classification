import torch

from config import get_config
from load_data import get_dataloader
from nets.try_cnns import TryCNNs

config = get_config()
_, _, test_dataloader = get_dataloader(config)
net = TryCNNs(config).load_state_dict(torch.load(''))
