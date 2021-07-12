import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("Fish.csv")

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.pairplot(data, hue = "Species").add_legend()
# plt.plot()
# print(data.Species)


## 이름 숫자로 변환
name = data.Species.unique()
for i, each in enumerate(name):
    data.loc[data['Species']==each, "Species"]= i
    # data[data['Species']==each].loc[0] = i

features = data.iloc[:,1:6]

target = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, stratify=target, random_state=0)


##################
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 7)
        self.dropout = nn.Dropout(0.2)
        self.batch = nn.BatchNorm1d(5)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.batch(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

class CustomDataset(Dataset):
  def __init__(self, x_train, y_train, transforms=None):
      self.x = x_train
      self.y = y_train


  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    x = self.x.iloc[idx].values
    y = self.y.iloc[idx]
    return x, y


# Data 준비
# X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
# y_train_tensor = torch.tensor(y_train.values.astype(np.int16))
train_dataset = CustomDataset(X_train, y_train, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# Model init
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.001)

# Modeling
best = 1
count = 0
saved_epoch = []
epoch = 1000
total_loss = 0
for i in range(epoch):
    total_loss = 0
    for t_x, t_y in train_loader:
        t_x = t_x.float().to(device)
        t_y = t_y.long().to(device)
        outputs = model(t_x)
        loss = criterion(outputs, t_y)
        optimizer.zero_grad()# 기울기 초기화
        loss.backward()# 가중치와 편향에 대해 기울기 계산
        total_loss += loss.item()

    optimizer.step()
    loss = total_loss / len(X_train)
    print(f"epoch -> {i}      loss -- > ", loss)

    if loss < best:
        torch.save(model.state_dict(), f"weight/model_{count}.pth")
        saved_epoch.append(i)
        count += 1
        # print("=====Saved!=====")

# Test1
test_dataset = CustomDataset(X_test, y_test, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

normal_acc = 50
for i in range(0,count):
    model.load_state_dict(torch.load(f'weight/model_{i}.pth'))
    model.eval()
    err = 0
    for t_x, t_y in test_loader:
        t_x = t_x.float().to(device)
        t_y = t_y.numpy()
        outputs = model(t_x)
        top = torch.topk(outputs, 1)
        top_index = top.indices.numpy()

        for y, t in zip(t_y, top_index):
            if y != t[0]:
                err += 1
    test_acc = int((len(X_test) - err)/len(X_test)  * 100)
    if test_acc > normal_acc:
        # print(f" weight/model_{i}.pth ::test acc = {test_acc}%")
        print(f" weight/model_{i}.pth :: epoch:{saved_epoch[i]} test acc = {test_acc}%")