# module import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

import numpy as np
import matplotlib.pyplot as plt

#장비 확인
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("device: ",DEVICE,", torch_version: ",torch.__version__)

BATCH_SIZE = 32
EPOCHS = 10

#datasets downloads
train_dataset = datasets.FashionMNIST(root="./data/FashionMNIST",
                                      train=True,
                                      download=True,
                                      transform=transforms.ToTensor())

test_dataset = datasets.FashionMNIST(root="./data/FashionMNIST",
                                      train=False,
                                      transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(datasets = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(datasets = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = False)

