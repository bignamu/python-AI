import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os

import urllib.request

URL = 'https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx'

driver = webdriver.Chrome('H:\chromedriver.exe')


driver.get(url=URL)

driver.implicitly_wait(3)


ygName = driver.find_elements_by_css_selector('td a')
ygG = driver.find_elements_by_css_selector('.asc')
ygH = driver.find_elements_by_css_selector('td:nth-child(9)')
ygHR = driver.find_elements_by_css_selector('td:nth-child(12)')
ygRBI = driver.find_elements_by_css_selector('td:nth-child(14)')

alist = []
def ele(ments):
    _list = []
    for e in ments:
        _list.append(e.text)

    return _list

alist.append(ele(ygName))
alist.append(ele(ygG))
alist.append(ele(ygH))
alist.append(ele(ygHR))
alist.append(ele(ygRBI))

print(alist)


x = len(alist[0])
y = len(alist)

    
with open('yg_rank.csv', 'w', encoding='utf-8') as file:
    file.write('이름,경기,안타,홈런,타점')
    for i in range(x):
        for j in range(y):
            file.write(f"{alist[j][i]},")
        file.write(f"\n")
driver.close()