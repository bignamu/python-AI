# 데이터 구조
# 4,921 개 학습용 데이터 / 테스트 1320 개

import itertools
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff # !pip install arff
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

# !pip install arff
# !pip install pandas
file_path = "./dataset"
train_fn = "FordA_TRAIN.arff"
test_fn = "FordA_TEST.arff"


def read_ariff(path):
    raw_data, meta = arff.loadarff(path)
    cols = [x for x in meta]

    data2d = np.zeros([raw_data.shape[0], len(cols)])

    for index, col in zip(range(len(cols)), cols):
        data2d[:, index] = raw_data[col]

    return data2d


train_path = os.path.join(file_path, train_fn)
test_path = os.path.join(file_path, test_fn)
train = read_ariff(train_path)
test = read_ariff(test_path)
print("train >> ", len(train))
print("test >>", len(test))

x_train_temp = train[:,:-1]
y_train_temp = train[:,-1] # 마지막 컬럼이 레이블 값

x_test = test[:,:-1]
y_test = test[:,-1]

print(x_test, y_test)

# 학습용 검증용 테스트용 데이터셋 나누기
normal_x = x_train_temp[y_train_temp==1] # train_x 테이터 중 정상 데이터
abnormal_x = x_train_temp[y_train_temp==-1] # train_x 데이터 중 비정상 데이터

normal_y = y_train_temp[y_train_temp==1]
abnormal_y = y_train_temp[y_train_temp==-1]

# 정상 데이터 8:2
# 정상 데이터를 8:2 나누기 위한 인덱스 설정
ind_x_normal = int(normal_x.shape[0]*0.8)
ind_y_normal = int(normal_y.shape[0]*0.8)
# 비정상 데이터 8:2
ind_x_abnoraml = int(abnormal_x.shape[0]*0.8)
ind_y_abnoraml = int(abnormal_y.shape[0]*0.8)


x_train = np.concatenate((normal_x[:ind_x_normal], abnormal_x[:ind_x_abnoraml]), axis=0) # 80
x_valid = np.concatenate((normal_x[ind_x_normal:], abnormal_x[ind_x_abnoraml:]), axis=0) # 20

y_train = np.concatenate((normal_y[:ind_y_normal], abnormal_y[:ind_y_abnoraml]), axis=0) # 80
y_valid = np.concatenate((normal_y[ind_y_normal:], abnormal_y[ind_y_abnoraml:]), axis=0)


# 데이터 확인
print("x_tain" , len(x_train))
print("x_valid" , len(x_valid))
print("y_train" , len(y_train))
print("y_valid" , len(y_valid))
print("x_test", len(x_test))
print("y_test", len(y_test))

# 시각화

# class 종류 정상 1 비정상 -1
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
print(classes)

x = np.arange(len(classes)) # plot x 축 개수
lables = ["Abnormal", "Normal"] # plot x 축 이름

# train, valid , test
valuse_train = [(y_train == i ).sum() for i in classes]
valuse_valid = [(y_valid == i ).sum() for i in classes]
valuse_test = [(y_test == i ).sum() for i in classes]

print(valuse_train, valuse_valid, valuse_test)


plt.figure(figsize = (8,4))
plt.subplot(1,3,1)
plt.title("Train_data")
plt.bar(x, valuse_train, width = 0.6, color=["red" , "blue"])
plt.ylim([0, 1500])
plt.xticks(x, lables)

plt.subplot(1,3,2)
plt.title("val_data")
plt.bar(x, valuse_valid, width = 0.6, color=["red" , "blue"])
plt.ylim([0, 1500])
plt.xticks(x, lables)

plt.subplot(1,3,3)
plt.title("test_data")
plt.bar(x, valuse_test, width = 0.6, color=["red" , "blue"])
plt.ylim([0, 1500])
plt.xticks(x, lables)
