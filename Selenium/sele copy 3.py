import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os

import urllib.request

URL = 'https://yandex.com/images/'

keyword = 'deep learning'
totalCount = 1

driver = webdriver.Chrome('H:\chromedriver.exe')


driver.get(url=URL)

driver.implicitly_wait(3)

divClick = driver.find_element_by_class_name('search2__input').click()
input_element = driver.find_element_by_name('text')

input_element.send_keys(keyword)
input_element.send_keys('\n')

images = driver.find_elements_by_class_name('serp-item__link')
time.sleep(0.5)

count = 0

for img in images:
    img.click()
    time.sleep(0.5)

    count += 1

    # imgURL = driver.find_element_by_xpath('/html/body/div[14]/div[1]/div/div/div[3]/div/div[2]/div[1]/div[3]/div/div/img')
    # imgURL = driver.find_element_by_css_selector('body > div.Popup2.Popup2_visible.Modal.Modal_visible.Modal_theme_normal.MMViewerModal.ImagesViewer.Theme.Theme_color_yandex-default.Theme_root_legacy > div.Modal-Table > div > div > div.MediaViewer.MediaViewer_theme_fiji.ImagesViewer-Container > div > div.MediaViewer-LayoutMain.MediaViewer_theme_fiji-LayoutMain > div.MediaViewer-LayoutScene.MediaViewer_theme_fiji-LayoutScene > div.MediaViewer-View.MediaViewer_theme_fiji-View > div > div > img')
    # imgURL = driver.find_element_by_class_name('MMImage-Origin')
    # imgURL = imgURL.get_attribute('href')
    # imgURL2 = driver.find_element_by_class_name('MMImageContainer')
    # imgURL2 = imgURL2.get_attribute('href')
    print(imgURL)
    urllib.request.urlretrieve(imgURL,keyword + str(count)+'.jpg')
    time.sleep(0.5)
    exit = driver.find_element_by_class_name('MMViewerModal-Close').click()
    time.sleep(0.5)
    print(count)
    if count == totalCount:
        break
    print('done')