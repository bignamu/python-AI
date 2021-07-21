from selenium import webdriver

URL = 'https://movie.naver.com/movie/point/af/list.nhn?&page=1'

driver = webdriver.Chrome('H:\chromedriver.exe')


driver.get(url=URL)

driver.implicitly_wait(3)

findTable = driver.find_elements_by_css_selector('.color_b')

for tb in findTable:
    test = tb.text
    print(test)

#old_content > table > tbody > tr:nth-child(1) > td.title > a.movie.color_b 

print(findTable)
''' 
with open('melon_rank.csv', 'w', encoding='utf-8') as file:
    file.write('순위,노래제목,가수')
    for i in data:
        file.write(f"{data[0]},{data[1]},{data[2]}")
 '''
driver.close()