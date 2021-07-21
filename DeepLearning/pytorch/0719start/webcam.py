import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import time
from copy import deepcopy # Add Deepcopy for args
import seaborn as sns 
import matplotlib.pyplot as plt

import glob
import os

import cv2

from torchvision import datasets, models, transforms
import PIL
from PIL import ImageFont, ImageDraw, Image
from torch.utils.data import Dataset, DataLoader, random_split

# device 설정 GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"

# data transforms
# image resiz -> 224 244
data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test' : transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
data_dir = 'H://Resources//pytorch_dataset//train_myhand'

parser = argparse.ArgumentParser()
args = parser.parse_args("")


args.train_batch_size = 16
args.test_batch_size = 16
args.epochs = 10


# 데이터 정의 및 데이터 loader 코드 구현
# data set (train, val, test)
train_data_set = torchvision.datasets.ImageFolder(data_dir,transform=data_transforms['train'])
# val_data_set = catdogDataset(data_dir, mode='val', transform=data_transforms['val'])
# test_data_set = catdogDataset(data_dir, mode="test", transform=data_transforms['test'])

# data loader
train_loader = DataLoader(train_data_set, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
# val_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False, drop_last=True)


# CNN Model (2 conv layers)
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))


        # Final FC 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(int(109/2) * int(109/2) * 64, 3, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc(out)
        return out

net = CNN()
# net = models.resnet50(pretrained=True).to(device)
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 3)  # make the change

net.load_state_dict(torch.load('H://Resources//AIschool/python-AI//DeepLearning//pytorch//0719start//2_47.37.pt',map_location=device))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001, weight_decay=5e-4)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print(len(train_loader))
epochLossdict = {}
epochAccdict = {}
valEpochLossdict = {}
valEpochAccdict = {}
for epoch in range(args.epochs):  # loop over the dataset multiple times
    net.train()

    lr_sche.step()
    correct = 0
    total = 0
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print statistics
        train_loss += loss.item()
        print(f'{epoch} ; {i}/{len(train_loader)} loss >> {loss / i} acc >> {correct / total}')

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total

    # # Check Validation
    # val_loss, val_acc = validate(net, partition, criterion, deepcopy(args))
    # valEpochLossdict[epoch] = val_loss
    # valEpochAccdict[epoch] = val_acc

    # Check Accuracy
    # acc = acc_check(resnet50, testloader, epoch, save=1)
    epochLossdict[epoch] = train_loss
    epochAccdict[epoch] = train_acc
    print(
        f'{epoch} epochs >> Accuracy train : {train_acc} val : val_acc ; Loss train : {train_loss} val : val_loss')

    if epoch:
        torch.save(net.state_dict(), "{}_{:2.2f}.pt".format(epoch, train_acc))

print('Finished Training')


# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)

def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    print(image)
    image = data_transforms['test'](image)
    image = image.float()
    #image = Variable(image, requires_autograd=True)
    image = image.to(device)
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image

fig,ax = plt.subplots()
loss = plt.plot(list(epochLossdict.keys()),list(epochLossdict.values()))
loss.xlabel('epoch')
loss.ylabel('loss')
loss.title('epoch / loss')

acc = plt.plot(list(epochAccdict.keys()),list(valEpochLossdict.values()))
acc.xlabel('epoch')
acc.ylabel('acc')
acc.title('epoch / acc')
loss.show()
acc.show()

print(list(epochLossdict.values()))

if not webcam.isOpened():
    print("Could not open webcam")
    exit()


# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    if not status:
        break
    img_original = frame.copy()
    img = preprocess(frame)


    # print(type(tf(img)))
    print(img.shape)
    outputs = net(img)
    _, predicted = torch.max(outputs.data, 1)


    print(predicted)
    result = predicted.item()

    if result == 0:
        me = "바위"
    elif result == 1:
        me = "보"
    elif result == 2:
        me = "가위"

    # display
    fontpath = "font/gulim.ttc"
    font1 = ImageFont.truetype(fontpath, 100)
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    draw.text((50, 50), me, font=font1, fill=(0, 0, 255, 3))
    frame = np.array(frame_pil)
    cv2.imshow('RPS', frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()

