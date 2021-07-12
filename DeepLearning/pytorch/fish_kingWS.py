import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_irisifsh
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
path = 'https://raw.githubusercontent.com/bignamu/python-AI/main/DeepLearning/0706_data/Fish.csv'
fish_data = pd.read_csv(path)

spec = list(set(fish_data['Species']))
spec.sort()
print(spec)
_dict = {}
for idx, fish in enumerate(spec):
  _dict[fish] = idx
print(_dict)
fish_data['Species'] = fish_data['Species'].map(_dict)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(6, 64)
        self.batch_norm1 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.7)

        self.fc2 = nn.Linear(64, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 7)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.silu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.silu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        # x = self.silu(x)

        return x
class CustomDataset(Dataset):
  def __init__(self, X_train, aa, transforms=None):
      self.X = X_train
      self.y = aa
      #self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.X)


  def __getitem__(self, idx):
    X = self.X[idx]
    y = self.y[idx]
    # print(type(X), type(y))
    # X = np.array(X)
    # y = np.array(y)
    return X, y

fish_features = fish_data[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
fish_label = fish_data['Species']



fish_features = fish_features.to_numpy()
fish_label = fish_label.to_numpy()
print(fish_features[0], fish_label[0])
print(type(fish_features),type(fish_label))
X_train, X_test, y_train, y_test = train_test_split(
                                                    fish_features,
                                                    fish_label,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=fish_label,
                                                    random_state=0
                                                    )
train_dataset = CustomDataset(X_train, y_train, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
device = torch.device('cpu')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.1)
epoch = 500

print(len(train_loader))
for i in range(epoch):
    total_loss = 0
    for X, y in train_loader:
        X = X.float().to(device)
        y = y.long().to(device)

        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 가중치와 편향에 대해 기울기 계산
        optimizer.step()
        total_loss += loss.item()



    if i % 100 == 0:
        torch.save(model.state_dict(), f"./model_{i}.pth")
    if i % 10 == 0:
        print(f"epoch -> {i}      loss -- > ", total_loss / len(X_train))

# 평가
model.eval()
model.load_state_dict(torch.load('model_400.pth'))
test_dataset = CustomDataset(X_test, y_test, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=127, shuffle=True)
err = 0

for t_x, t_y in test_loader:
    t_x = t_x.float().to(device)
    t_y = t_y.long().to(device)
    outputs = model(t_x)
    correct_prediction = torch.argmax(outputs, 1) == t_y
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())