import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os

import urllib.request

URL = 'https://www.google.co.kr/imghp?hl=ko'

keyword = 'akakubi'
totalCount = 3

driver = webdriver.Chrome('H:\chromedriver.exe')


driver.get(url=URL)

driver.implicitly_wait(3)

input_element = driver.find_element_by_name('q')
input_element.send_keys(keyword)
input_element.send_keys(Keys.RETURN)

images = driver.find_elements_by_css_selector('.Q4LuWd')

count = 0

for img in images:
    img.click()
    time.sleep(0.5)

    count += 1

    imgURL = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img')
    urllib.request.urlretrieve(imgURL,'./'+str(keyword)+str(count)+'.jpg')

    if count == totalCount:
        break
    print('done')