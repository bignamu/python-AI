import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.preprocessing import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(6, 4096)
        self.fc2 = nn.Linear(4096, 72)
        self.fc3 = nn.Linear(72,7)
        self.batch_norm1 = nn.BatchNorm1d(7)
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(3, 3)
        
        self.sig = nn.Sigmoid()
        

    def forward(self, x):
        x = self.fc1(x)   
        x = self.relu(x)     
        x = self.fc2(x)
        #x = self.relu(x)
        x = self.fc3(x)
        # x = self.batch_norm1(x)
        # x = self.relu(x)
        
        
        # x = self.fc2(x)
        # x = self.sig(x)
        
        return x


class CustomDataset(Dataset):
    def __init__(self, data1, data2, transforms=None):
        self.x = data1
        self.y = data2
        # print(self.x) 
        # print(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = np.array(x)
        y = np.array(y)
        return x, y


Fish_Data= pd.read_csv('https://raw.githubusercontent.com/bignamu/python-AI/main/DeepLearning/0706_data/Fish.csv', sep='\t')  #pandas로 데이터 불러오기 구분자가 탭으로 되어있어서 sep="\t" 추가함
#불러온 데이터를 확인해보니 물고기 종류는 7가지(label 값),  Feature 값은 총 6개로 확인 할 수 있었음
print("Fish_Data : \n", Fish_Data)

Fish_Species=Fish_Data['Species']  # 물고기 종류에 해당하는 컬럼만 가져옴
Fish_Species_list=[]  #현재 Fish_Species는 Object형식으로 저장되어 있어서 데이터 가공이 매우 어려움 따라서 그것을 각각 개별적으로 저장하여 list화 시킬예정
for i in range(0,159):
    Fish_Species_list.append(Fish_Species.iloc[i]) # 각 row에 있는 물고기 이름의 데이터들을 for문을 돌며 가져와서 list에 추가시킴

# for문을 돌면서 각 Feature값들을 나중에 학습시키기 위해 0~6까지의 숫자로 바꿈
for i in range(0, 159):                 
    if Fish_Species_list[i] == "Bream":
        Fish_Species_list[i]  = 0        
    elif Fish_Species_list[i] == 'Roach':
        Fish_Species_list[i] = 1
    elif Fish_Species_list[i] == 'Whitefish':
        Fish_Species_list[i] =2
    elif Fish_Species_list[i]== 'Parkki':
        Fish_Species_list[i] =3
    elif Fish_Species_list[i] == 'Perch':
        Fish_Species_list[i]= 4
    elif Fish_Species_list[i]== 'Pike':
        Fish_Species_list[i]= 5
    else : 
        Fish_Species_list[i]= 6

Fish_Species_list=np.array(Fish_Species_list) #리스트로 저장하여 완성된 데이터를 넘파이를 이용하여 배열로 바꿈(나중에 학습시킬 때 텐서화 시키기 편하게 하기 위함)
print("Fish_Species_list : \n", "Bream=0, Roach=1, Whitefish=2, Parkki=3, Perch=4, Pike=5, Smelt=6 \n", Fish_Species_list)

Fish_feature = DataFrame(Fish_Data, columns=['Weight', 'Length1','Length2','Length3','Height','Width'])
print("Fish_feature : \n",Fish_feature)
# print("###############################################################################")
Fish_feature = StandardScaler().fit_transform(Fish_feature) ###########################

print("Fish_feature : \n",Fish_feature)

#Fish_feature.shape = (159,6)

# Fish_feature_list=[]
# for i in range(0,159):
#     Fish_feature_list.append(Fish_feature.iloc[i].to_list()) #각 row의 Feature값들을 한줄 한줄 가져와서 Feature list에 저장

# print(Fish_feature_list)   
# # Fish_feature_list=np.array(Fish_feature_list) # 역시나 넘파이를 이용하여 배열로 바꿈
# print("Fish_feature_list : \n",Fish_feature_list)

#########
#결과 Fish_Species_list : 물고기 종류가 0~6까지의 숫자로써 넘파이 형태의 label data가 들어있음
#결과 Fish_feature_list : 물고기의 feature data가 넘파이 형태로 저장되어 있음
#########

x_train, x_test, x_traintarget, x_testtarget = train_test_split(Fish_feature, Fish_Species_list, test_size=0.25, shuffle = True, stratify = Fish_Species_list, random_state=100)

train_dataset = CustomDataset(x_train, x_traintarget, transforms=None) 
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
model = Net().to(device) 
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.01) 
epoch = 750  # 학습을 몇번 돌릴건지 결정
total_loss = 0  
model.train() 

for i in range(epoch):
    for x, y in train_loader:
        x = x.float().to(device)
        y = y.long().to(device)
        # print("x : \n", x)
        # print("y : \n", y)                 
        
        outputs = model(x)
        # print("x : \n", x)
          # model x를 호출함과 동시에 Net의 forward가 실행되면서 변화가 일어남
        # print(outputs)
        # print("outputs : ", outputs)
        # print("y:  ", y) 
        loss = criterion(outputs, y)
        # print(loss)
        # outputs = outputs.detach().numpy()
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 가중치와 편향에 대해 기울기 계산
        total_loss += loss.item() #loss값이 텐서값이 나오므로 그 안의 숫자만 받겠다는 뜻
        outputs = outputs.detach().numpy()
        y = y.numpy()
           
        # print(outputs)
        # print(y)
    if i % 50 == 0:
        torch.save(model.state_dict(), f"fish_classification_model_last.pth")
        print(f"epoch -> {i}      loss -- > ", total_loss / len(train_loader))
    
    optimizer.step()
    total_loss = 0

model.eval()  
model.load_state_dict(torch.load('fish_classification_model_last.pth'))


test_dataset = CustomDataset(x_test,x_testtarget, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
for x, y in test_loader:
    x = x.float().to(device)
    y = y.long().to(device)
    # print(x, y)
    outputs = model(x)
    # print(outputs, y)

count = 0
for x, y in test_loader:
    x = x.float().to(device)
    y = y.long().to(device)
    #print(x, y)
    outputs = model(x)
    #print(x,y,outputs)
    value = outputs.detach().numpy()
    idx = np.argmax(value)
    #print(y, idx)
    if y == idx:
        count += 1
#print(count)
print(f'{count/len(x_testtarget)*100}%')