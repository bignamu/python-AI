import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import torch.nn.functional as F



## 설명 코드
class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
    
            self.fc1 = nn.Linear(4, 64)
            self.fc2 = nn.Linear(64, 128)
            self.fc3 = nn.Linear(128, 3)
    
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
    
            return x


class CustomDataset(Dataset):
  def __init__(self, x_train, y_train, transforms=None):
      self.x = x_train
      self.y = y_train


  def __len__(self):
    return len(self.x)


  def __getitem__(self, idx):
    X = self.x[idx]
    y = self.y[idx]
    return X,y


iris = load_iris()
iris_data = iris.data
iris_label = iris.target
print(type(iris_data))
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                    test_size=0.2,shuffle=True, stratify=iris_label, random_state=0)

train_dataset = CustomDataset(x_train, y_train, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.1)
epoch = 100
total_loss = 0
print(len(train_loader))
for i in range(epoch):
    total_loss = 0
    for t_x, t_y in train_loader:
        t_x = t_x.float().to(device)
        t_y = t_y.long().to(device)
        print(t_x, t_y)
        print(t_x.shape,t_y.shape)
        print(t_x.dim(),t_y.dim())

        outputs = model(t_x)
        loss = criterion(outputs, t_y)
        print(t_y)
        exit()

        optimizer.zero_grad()# 기울기 초기화
        loss.backward()# 가중치와 편향에 대해 기울기 계산
        total_loss += loss.item()

    optimizer.step()
    print(f"epoch -> {i}      loss -- > ", total_loss / len(x_train))
    if i % 20 == 0:
        torch.save(model.state_dict(), f"weight/model_{i}.pth")

test_dataset = CustomDataset(x_test, y_test, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
model.load_state_dict(torch.load('weight/model_60.pth'))
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
print(f"test acc = {int((len(x_test) - err)/len(x_test)  * 100)}%")