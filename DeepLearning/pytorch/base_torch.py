import torch

# https://wikidocs.net/57165 CustomDataset

import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self,data,transform=None):
        self.x = [i[0] for i in data]
        self.y = [i[1] for i in data]
        # 데이터셋의 전처리를 해주는 부분


    def __len__(self):
        return len(self.x)
        # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분


    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y
        # 데이터셋에서 특정 1개의 샘플을 가져오는 함수


data = [[2,0],[4,0],[6,0],[8,1],[10,1],[12,1]]

train_dataset = CustomDataset(data,transform=None)
train_loader = DataLoader(train_dataset,batch_size=2,shuffle=False)

for x, y in train_loader:
    print(x,y)
