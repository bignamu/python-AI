import cv2
import numpy as np
import json
import os
import pandas as pd


# 이미지읽기
img = cv2.imread('../0706_data/dog.jpg')
c_img = cv2.imread('../0706_data/cat.jpg')

# 너비, 높이, RGB(색상)
print(img.shape)

# 이미지 저장
# cv2.imwrite('copy_img.jpg',img)

# cv2.imshow('dog',img)
# cv2.waitKey()

# 색 변화 컨버터 cv2는 BGR로 읽힌다 색깔범위 0~255
# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('',rgb_img)
# cv2.imshow('g',gray_img)
# cv2.waitKey()

# 색으로 보기 이런게있다 정도만

# (B, G, R) = cv2.split(img)
#
# color = R
# cv2.imshow("",color)
# cv2.waitKey()
#
# zeros = np.zeros(img.shape[:2],dtype='uint8')
#
# cv2.imshow('Red',cv2.merge([zeros,zeros,R]))
# cv2.imshow('Green',cv2.merge([zeros,G,zeros]))
# cv2.imshow('Blue',cv2.merge([B,zeros,zeros]))
# cv2.waitKey(0)

# 픽셀 값 접근
# print(img[100,200])
# cv2.imshow('',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

'''자르기 사각형 정도만'''

# 크기조절
# cv2.imshow('',img)
# img = cv2.resize(img,(408,380))
# cv2.imshow('big',img)
# img = cv2.resize(img,(108,58))
# cv2.imshow('small',img)
# cv2.waitKey()

# 자르기
# cv2.imshow('',img[0:150,0:180])
#
# cv2.imshow('change',img[100:150,50:100])
# h, w, c = img.shape
#
# cv2.imshow('crop',img[int(h/2-50): int(h/2+50),int(w/2-50):int(w/2+50)])
# print(int(h/2-50),int(h/2+50),int(w/2 - 50),int(w/2 +50))
# cv2.waitKey()

# 도형 그리기

# line
# 시작점,끝점,rgb값, 선굵기 //  BGR인것을 인지
# img = cv2.line(img,(100,100),(180,150),(0,255,0),4)
# cv2.imshow('',img)
# cv2.waitKey()


# rectangle
# 시작점,끝점,rgb값, 선굵기 //  BGR인것을 인지
# img = cv2.rectangle(img,(35,26),(160,170),(0,255,0),3)
# cv2.imshow('',img)
# cv2.waitKey()


# circle
# # 시작점,반지름,rgb값, 선굵기 //  -1이 꽉찬 원
# img = cv2.circle(img,(200,100),30,(0,255,0),3)
# cv2.imshow('',img)
# cv2.waitKey()

# # poly
# 다각형 거의 안씀
# pts = np.array([[35,26],[35,170],[160,170],[190,26]])
# img = cv2.polylines(img,[pts],True,(0,255,0),3)
# cv2.imshow('',img)
# cv2.waitKey()


# text 텍스트 넣기 // 시작점 폰트종류 글자크기 글자굵기
# img = cv2.putText(img,'dog',(200,100),0,1,(0,25,0),2)
# cv2.imshow('',img)
# cv2.waitKey()

# 이미지 붙여넣기
# img = cv2.rectangle(img,(200,100),(275,183),(0,255,0),2)
#
#
# c_img = cv2.resize(c_img,(75,83))
# img[100:183,200:275] = c_img
# cv2.imshow('change',img)
# cv2.waitKey()

# 이미지 더하기
# img = cv2.resize(img,(217,232))
# add1 = img + c_img
# add2 = cv2.addWeighted(img,float(0.8),c_img,float(0.2),5) # 이미지 비중
# cv2.imshow('1',add1)
# cv2.imshow('2',add2)
# cv2.waitKey()

'''자주 사용하는 기능'''
# 이미지 회전
# height, width, c = img.shape
# img90 = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) #시계방향 90도
# img270 = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계 90도
# img180 = cv2.rotate(img,cv2.ROTATE_180)
#
#
# # 중심점
# img_r = cv2.getRotationMatrix2D((width/2,height/2),45,1)
# cv2.imshow('90',img90)
# cv2.imshow('2',img270)
# cv2.imshow('3',img180)
# cv2.imshow('4',img_r)
# cv2.waitKey()




# 이미지 반전 flip 0=상하대칭 1=좌우대칭 좌우대칭을 많이 쓴다더라
# cv2.imshow('origin',img)
# img = cv2.flip(img,0)
# cv2.imshow('270',img)
# cv2.waitKey()



# 이미지 아핀
# height, width, channel = img.shape
# matrix = cv2.getRotationMatrix2D((width/2,height/2),45,2)
# img = cv2.warpAffine(img,matrix,(width,height))
# cv2.imshow('270',img)
# cv2.waitKey()



# 이미지 밝기, 감마
# nimg = cv2.imread('../0706_data/night.jpg')
# table = np.array([((i/255.0)**0.5) * 255 for i in np.arange(0,256)]).astype('uint8')
# gamma_img = cv2.LUT(nimg, table)
#
# val = 50 #randint(10,50)
# # numpy.full 함수는 주어진 형태와 타입을 갖는, 주어진 값으로 채워진 새로운 어레이를 반환합니다.
# # numpy.arange 함수는 주어진 간격에 따라 균일한 어레이를 생성합니다.
# array = np.full(nimg.shape, (val,val,val),dtype=np.uint8) # 픽셀당 50이 더해진 것
# all_array = np.full(nimg.shape,(30,30,30),dtype=np.uint8) # 픽셀당 30이 더해진 것
# bright_img = cv2.add(nimg,array).astype('uint8')
# all_img = cv2.add(gamma_img,all_array).astype('uint8')
#
# cv2.imshow('origin',nimg)
# cv2.imshow('all',all_img)
# cv2.imshow('bright',bright_img)
# cv2.imshow('gamma',gamma_img)
# cv2.waitKey()



# # 이미지 블러링 개인정보때문에 중요성이 높다
# blu_img = cv2.blur(img,(15,15))
#
# roi = img[28:74,95:165] # blur할 부분
# cv2.imshow('blurLoca',roi)
# cv2.waitKey()
#
#
# roi = cv2.blur(roi,(15,15)) #
# img[28:74,95:165] = roi
# cv2.imshow('blu',blu_img)
# cv2.imshow('s-blu',img)
# cv2.waitKey()

# # 이미지 패딩 정사각형으로 놓을경우가 많다 자주씀
# img_pad = cv2.cv2.copyMakeBorder(img,50,50,100,100,cv2.BORDER_CONSTANT, value=[0,0,0]) # top bottom left right , 중심에 놓겠다 , 검은색 패딩하겠다
# cv2.imshow('img pad',img_pad)
# cv2.waitKey()


# cv2 cascade 거의 안씀

# face_cascade = cv2.CascadeClassifier('../0706_data/haarcascade_frontalface_default.xml')
#
# img = cv2.imread('../0706_data/face.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray, 1.3, 1)
#
# print(faces)
#
# cv2.imshow('face', img)
# cv2.waitKey()

#
# total_list = []
# for i in range(5):
#     ins_dic = {}
#     ins_dic[f'person{i}'] = i
#     ins_dic['bbox'] = [i+5,i+10,i+15,i+30]
#     total_list.append(ins_dic)
#
# with open('json_sample.json','w',encoding='utf-8') as make_file:
#     json.dump(total_list,make_file,indent='\t')
#
#
# json_dir = 'json_sample.json'
# print(os.path.isfile(json_dir))
# with open(json_dir) as f:
#     json_data = json.load(f)
#
# for j_data in json_data:
#     print(j_data)
#     print(j_data['bbox'])