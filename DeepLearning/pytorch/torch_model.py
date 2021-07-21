import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 1)
        # self.batch_norm1 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        print(x.shape)
        print(x)
        x = self.fc1(x)
        print(x.shape)
        print(x)
        # x = self.batch_norm1(x)
        # x = self.relu(x)
        # # x = self.dropout(x)
        # x = self.fc2(x)
        x = self.relu(x)
        print(x.shape)
        print(x)

        return x



class CustomDataset(Dataset):
  def __init__(self, data, transforms=None):
      self.x = [i[0] for i in data]
      self.y = [i[1] for i in data]


  def __len__(self):
    return len(self.x)


  def __getitem__(self, idx):
      x = [self.x[idx]]
      y = self.y[idx]
      x= np.array(x)


      return x, y


torch.manual_seed(1)
data = [[2,0], [4,0], [6,0], [8,1], [10,1], [12,1]]

train_dataset = CustomDataset(data, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)

for x, y in train_loader:

    x = x.float().to(device)
    y = y.float().to(device)
    print(x)
    print('------------------'*5)
    outputs = model(x)
    print('------------------'*5)
    print(outputs)
    exit()
    outputs = outputs.detach().numpy()
    print(outputs)
    print(y)
    exit()