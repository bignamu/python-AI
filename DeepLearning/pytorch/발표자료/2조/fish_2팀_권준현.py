# 0709 - 조별실습

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(6, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc_final = nn.Linear(16, 7)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)

        x = self.fc2(x)
        x = func.relu(x)

        x = self.fc3(x)
        # x = self.fc4(x)
        # x = self.fc5(x)

        x = self.fc_final(x)
        x = self.sig(x)

        return x


class CustomDataset(Dataset):
    def __init__(self, data, label, transforms=None):
        self.x = data
        self.y = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = np.array(x)
        y = np.array(y)

        return x, y


# [데이터 로드]
fish_header = []
fish_data = []
with open('Fish.csv', mode='r', encoding='utf-8-sig') as fish_csv:
    rdr = csv.reader(fish_csv)
    header = True
    for line in rdr:
        if header:
            for item in line:
                fish_header.append(item)
                header = False
        else:
            fish_data.append(line)

# print(fish_header)
# print('----')
# print(fish_data[:4])

# [데이터 전처리]
fish_data = np.array(fish_data)

# # feature/label 분리
fish_X = fish_data[:, 1:]
fish_X = np.array(fish_X, dtype=float)
fish_y = fish_data[:, 0]

# feature 부호화
species_set = set()
for sp in fish_y:
    species_set.add(sp)  # 중복제거

species_list = list(species_set)
species_list.sort()
fish_y = np.array([species_list.index(sp) for sp in fish_y])
print(species_list)

plt.rc('font', family='SeoulNamsan')
# 상관관계: 무게 vs 종
plt.xlabel('weight')
plt.scatter(fish_X[:, 0], fish_y, marker='o', alpha=0.4)
plt.yticks(list(range(7)), species_list)
plt.show()

# 상관관계: 길이 vs 길이 by 종
sc = plt.scatter(x=fish_X[:, 2] - fish_X[:, 1], y=fish_X[:, 3] - fish_X[:, 2], marker='o', s=10, alpha=0.4, c=fish_y,
                 cmap=plt.cm.get_cmap('rainbow', 7))
plt.xlabel('L2 - L1')
plt.ylabel('L3 - L2')
plt.legend(*sc.legend_elements())
plt.show()

# 상관관계: 너비 vs 높이 by 종
sc = plt.scatter(x=fish_X[:, 5], y=fish_X[:, 4], marker='o', s=10, alpha=0.4, c=fish_y,
                 cmap=plt.cm.get_cmap('rainbow', 7))
plt.xlabel('width')
plt.ylabel('height')
plt.legend(*sc.legend_elements())
plt.show()

x_train, x_test, y_train, y_test = \
    train_test_split(fish_X, fish_y, test_size=1 / 5, random_state=20210708, shuffle=True, stratify=fish_y)

if torch.cuda.is_available():
    print('Using CUDA')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# [선언]
model = Net().to(device)
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=256, gamma=0.2)

# [학습]
max_epoch = 4096
save_step = 50
train_batch_size = 24
train_dataset = CustomDataset(data=x_train, label=y_train)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
train_loss_dict = {}

os.makedirs('./models', exist_ok=True)
do_train = True  # False 시 학습 생략)
if do_train:
    os.system('echo y | del models')
    print('Waiting for files to be deleted...')
    time.sleep(5)
    model.train()  # 학습 모드
    for i in range(max_epoch):
        total_loss = 0
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.long().to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            outputs = outputs.detach()
            outputs = outputs.numpy()
            y = y.numpy()

        avg_loss = total_loss / len(x_train)
        if i % save_step == 0:
            print(f"epoch -> {i:>4}        loss -- > ", avg_loss)
            torch.save(model.state_dict(), f'./models/fish_model_{i:04d}.pth')  # 모델 저장
        elif i == max_epoch - 1:
            print(f"epoch -> {i:>4}(final) loss -- > ", avg_loss)
            torch.save(model.state_dict(), './models/fish_model_last.pth')  # 모델 저장

        train_loss_dict[i] = avg_loss
        optimizer.step()


# [평가]
def test_model(model, criterion, saved_model, x_test, y_test):
    model.eval()  # 평가 모드
    model.load_state_dict(torch.load(saved_model))  # 저장된 모델 로드
    test_dataset = CustomDataset(data=x_test, label=y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    score = 0
    total_loss = 0
    for x, y in test_loader:
        x = x.float().to(device)
        y = y.long().to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.item()

        y = y.numpy()
        top = torch.argmax(outputs, dim=1)  # 가장 높은 값의 index
        # print(top, y)
        # print(top_idx)
        for _y, t in zip(y, top):
            if _y == t:
                score += 1

    accuracy = score / len(x_test)
    avg_loss = total_loss / len(x_test)

    return accuracy, avg_loss


test_accu_dict = {}
test_loss_dict = {}

for saved_model in os.listdir('./models'):
    result = test_model(model, criterion, f'./models/{saved_model}', x_test, y_test)
    print(saved_model, f'accuracy {result[0] * 100:>.3f}%, avg loss {result[1]:>.6f}')
    # fish_model_last.pth
    i = saved_model[11:15]
    if i == 'last':
        i = max_epoch - 1
    else:
        i = int(i)
    test_accu_dict[i] = result[0]
    test_loss_dict[i] = result[1]
    # exit()

# 결과발표
train_loss_x, train_loss_y = list(train_loss_dict.keys()), list(train_loss_dict.values())
test_loss_x, test_loss_y = list(test_loss_dict.keys()), list(test_loss_dict.values())
test_accu_x, test_accu_y = list(test_accu_dict.keys()), list(test_accu_dict.values())
test_accu_y = [y * 100 for y in test_accu_y]

# max accu
accu_max = -1
epoch_at_accu_max = []

for ep, ac in zip(test_accu_x, test_accu_y):
    if ac > accu_max:
        accu_max = ac
        epoch_at_accu_max = [ep]
    elif ac == accu_max:
        epoch_at_accu_max.append(ep)

print('----')
print(f'max accuracy: {accu_max}% @ Epoch{"" if len(epoch_at_accu_max) == 1 else "s"} {epoch_at_accu_max}')

# plot
fig, ax1 = plt.subplots()
fig.suptitle(f'{max_epoch} epochs, {train_batch_size} batch-size, {accu_max}% max accuracy')
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')

ax1.set_ylabel('loss', color='k')
ax1.plot(train_loss_x, train_loss_y, 'g-', linewidth=2, label='Train loss')
ax1.plot(test_loss_x, test_loss_y, 'r-', linewidth=2, label='Test loss')

ax2.set_ylim(-5, accu_max + 5)
ax2.set_ylabel('accuracy (%)', color='b')
ax2.plot(test_accu_x, test_accu_y, 'b--', linewidth=1, label='Test accuracy(%)')
for ep in epoch_at_accu_max:
    ax2.scatter(ep, accu_max, marker='*', color='gold')
    # ax2.text(ep + 5, accu_max, ep, fontsize=10)

ax1.legend(loc='best')

plt.show()
