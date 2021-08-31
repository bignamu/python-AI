import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from konlpy.tag import Mecab
from nltk import FreqDist
from eunjeon import Mecab



excelpath = "H:\\Resources\\pytorch_dataset\\안녕_미안.xlsx"

dataset = pd.read_excel(excelpath)

dataset = dataset.drop(dataset.columns[2:],axis=1)
print(dataset.head())


dataset['x'] = dataset['x'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")


stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


tokenizer = Mecab()
tokenized=[]
for sentence in dataset['x']:
    temp = tokenizer.morphs(sentence) # 토큰화
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    tokenized.append(temp)

print(tokenized[:10])

vocab = FreqDist(np.hstack(tokenized))
print('단어 집합의 크기 : {}'.format(len(vocab)))



