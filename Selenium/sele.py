from selenium import webdriver


''' 

URL = 'https://www.miraeassetdaewoo.com/hki/hki3028/r01.do'

driver = webdriver.Chrome(executable_path='chromedriver')

driver.get(url=URL)

driver.implicitly_wait(time_to_wait=5)



 '''




driver = webdriver.Chrome('H:\chromedriver.exe')
driver.get('https://www.melon.com/chart/day/index.htm')



driver.implicitly_wait(3)

#frm > div > table > tbody

#list_song = driver.find_elements_by_css_selector('#frm > div > table > tbody')
list_song = driver.find_elements_by_css_selector('.service_list song table tbody tr')



data = []
rank = 1

for tr in list_song:
    title = tr.find_element_by_css_selector('.wrap_song_info .ellipsis.rank01').text
    singer = tr.find_element_by_css_selector('.wrap_song_info .rank02').text
    data.append([rank, title, singer])
    rank += 1
    print(title, singer)    

with open('melon_rank.csv', 'w', encoding='utf-8') as file:
    file.write('순위,노래제목,가수')
    for i in data:
        file.write(f"{data[0]},{data[1]},{data[2]}")

driver.close()