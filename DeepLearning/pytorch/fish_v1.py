import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


# 사용할 모듈 생성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 7)                 # 깊이는 1 ~ 2
        # self.fc2 = nn.Linear(16, 256)             # 참고 : https://channelofchaos.tistory.com/105
        # self.fc3 = nn.Linear(256, 7)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.sig(x)
        # x = self.dropout(x)
        #
        # x = self.fc3(x)
        # x = self.relu(x)

        return x


class CustomDataset(Dataset):
    def __init__(self, x, y, transforms=None):
        self.x = [i for i in x]
        self.y = [i for i in y]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]         # x, y의 차원을 맞춰야한다.
        y = self.y[idx]         # 출력을 통해서 확인해보자
        x = np.array(x)
        y = np.array(y)

        return x, y



# 주어진 csv 파일 읽기
fish_file = pd.read_csv('Fish.csv')

# fish_file의 기본적인 정보 확인
# print(type(fish_file))
# print(fish_file.shape)
# print(fish_file)

# 전반적인 데이터 살펴보기
# 1. 값에 대한 정보 확인
# print(fish_file.describe())
# 2. 각 변수들의 상관관계 파악 / Length간의 관계파악을 위해 사용
# sns.pairplot(fish_file, hue='Species')          # Length끼리 거의 일직선상 관계이므로 하나로 사용도 해볼 것
# plt.show()                                      # Perch / Roach / Whitefish (보라/주황/초록) 의 분포가 비슷해보이므로 확인
# 3. Perch / Roach / Whitefish 의 분포를 확인
# 데이터 양의 차이가 있는 것으로 보임
# print('----- Perch -----')                                        # 전반적인 데이터 양의 차이가 있음
# print(fish_file[fish_file['Species'] == 'Perch'].describe())      # 살짝 걸쳐서 만나는 정도
# print('----- Roach -----')                                        # Length를 하나만 쓰면 거의 안겹칠 듯
# print(fish_file[fish_file['Species'] == 'Roach'].describe())      # 같은 어류에 대해서 Length가 비슷해도
# print('----- Whitefish -----')                                    # Weight, Height, Width에서 차이가 발생할 것
# print(fish_file[fish_file['Species'] == 'Whitefish'].describe())



# 모듈에 넣기위해 데이터 가공
cols = fish_file.columns.tolist()   # Features
cols.remove('Species')              # '종'은 알아내야되는 정보이므로 따로 배열에 넣기위해 제거

features, species = [], []          # Feature의 값을 넣을 배열 / '종'을 넣을 배열
for i in range(159):
    features.append([])
    for feat in cols:
        features[i].append(fish_file[feat][i])
    species.append(fish_file['Species'][i])

speciesSet = list(set(species))     # 몇 종인지 확인하기 위해 set으로 중복 삭제
speciesDict = dict(zip(speciesSet, [i for i in range(len(speciesSet))]))      # species의 값을 숫자로 바꾸기 위한 dict
for i in range(len(species)):       # species의 문자 -> 숫자
    species[i] = speciesDict[species[i]]

# print(features)                     # 값 확인
# print(species)



# train/test data로 분류
x_train, x_test, y_train, y_test = train_test_split(features, species, test_size=0.2,
                                                    shuffle=True, stratify=species, random_state=3)

# 모듈에 가져온 데이터 넣기
# dataset & loader 설정
batch_size = 3
train_dataset = CustomDataset(x_train, y_train, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)    # BatchNorm1d을 활성화 시키면
                                                                                        # batch_size > 1
# device & model 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)

# criterion & optimizer & scheduler 설정
criterion = nn.CrossEntropyLoss()           # sigmoid + softmax
# 0.03 - 1.82 / 0.01 - 1.8(3~6) / 0.1 - 1.8(0~3)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer = optim.Adam(model.parameters(), lr=0.03)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 학습
epoch = 1000
total_loss = 0
min_loss = 100
for i in range(epoch):
    for x, y in train_loader:
        x = x.float().to(device)
        y = y.long().to(device)     # float으로 설정 시 오류 발생
                                    # expected scalar type Long but found Float
        # print(x, y)
        outputs = model(x)
        # print(outputs, y)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()

        total_loss += loss.item()
        outputs = outputs.detach().numpy()
        y = y.numpy()

    if (total_loss/len(x_train)) < min_loss:
        min_loss = total_loss
        torch.save(model.state_dict(), f"opt_model_v1.pth")

    if i % 50 == 0:
        print(f"epoch -> {i}      loss -- > ", total_loss / len(x_train))
    optimizer.step()
    total_loss = 0
    model.train()

# 평가
model.eval()
model.load_state_dict(torch.load('opt_model_v1.pth'))
test_dataset = CustomDataset(x_test, y_test, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
totalCount = 0
trueCount = 0
for x, y in test_loader:
    x = x.float().to(device)
    y = y.long().to(device)
    outputs = model(x)

    # 정확도 계산
    for i in range(batch_size):
        try:                # 개수에 문제가 있어서 try - except 구문 적용
            totalCount += 1
            trueY = y.detach().numpy()[i]
            predY = outputs.detach().numpy()[i]
            for j in range(7):
                if predY[trueY] < predY[j]:
                    break
                if j == 6:
                    trueCount += 1
        except:
            break

print(trueCount / totalCount)
