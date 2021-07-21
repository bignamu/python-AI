import cv2
import numpy as np
import json
import os
import pandas as pd


# 실습 1

# face_cascade = cv2.CascadeClassifier('../0706_data/haarcascade_frontalface_default.xml')
#
# img = cv2.imread('../0706_data/face.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray, 1.3, 1)
#
# print(faces)
#
# _list = list(faces)
# cnt = 0
# while _list:
#     x_pos = []
#     y_pos = []
#     for idx in _list:
#         for j in range(len(idx)):
#             if j < 2:
#                 x_pos.append(idx[j])
#             else:
#                 y_pos.append(idx[j])
#         img = cv2.rectangle(img, (x_pos[0], x_pos[1]), (x_pos[0]+y_pos[0],x_pos[1]+y_pos[1]), (0, 255, 0), 3)
#         cnt += 1
#         print(cnt, idx)
#         _list.pop(0)
#         break
#
#
#
#
# img = cv2.rectangle(img,(5,20),(69,69),(0,255,0),3)
# cv2.imshow('',img)
# cv2.waitKey()
#


#실습 2

img = cv2.imread('../0706_data/test_img3.jpg')
print(img.shape)

height = img.shape[0]
width = img.shape[1]

h = int((500-height)/2)
w = int((500-width)/2)

img_pad = cv2.cv2.copyMakeBorder(img,h,h,w,w,cv2.BORDER_CONSTANT, value=[0,0,0]) # top bottom left right , 중심에 놓겠다 , 검은색 패딩하겠다
print (img_pad.shape)


cv2.imshow('img pad',img_pad)
cv2.waitKey()


pad_cas = cv2.CascadeClassifier('../0706_data/haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)

faces = pad_cas.detectMultiScale(gray, 1.3, 1)

print(faces)

_list = list(faces)
cnt = 0
dicli = []
while _list:
    _dict = {}

    for idx in _list:

        _dict['object'] = 'person{}'.format(cnt)
        _dict['box'] = list(idx)
        dicli.append(_dict)
        _list.pop(0)
        cnt += 1
        break



print(dicli)
#
#
# # 실습 3
#
#
# img = cv2.imread('../0706_data/test_img3.jpg')
#
# for dic in dicli:
#     face_pos = dic['box']
#     print(face_pos)
#     img = cv2.cv2.copyMakeBorder(img,h,h,w,w,cv2.BORDER_CONSTANT, value=[0,0,0]) # top bottom left right , 중심에 놓겠다 , 검은색 패딩하겠다
#
# cv2.imshow('json to pad',img)
# cv2.waitKey()