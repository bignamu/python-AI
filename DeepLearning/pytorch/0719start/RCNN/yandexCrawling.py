from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from argparse import ArgumentParser
import urllib.request
import os
from selenium.webdriver.chrome.options import Options
options = Options()
# options.add_argument('--kiosk')
options.add_argument("--start-maximized")

parser = ArgumentParser(description='yandex')
parser.add_argument('--count',default = 1000)
parser.add_argument('--query',default = 'yandex_mask')

args = parser.parse_args()
# print(args.count)


# exit()
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)




def run(query, count):
	
	createFolder('./'+query+'/')

	browser = webdriver.Chrome('H:\chromedriver.exe',options=options)
	# browser.set_window_size(1024, 768)


	# url = 'https://yandex.com.tr/gorsel/search?text='+query
	url = 'https://yandex.com/images/search?url=https%3A%2F%2Favatars.mds.yandex.net%2Fget-images-cbir%2F2271911%2FqGMDLJbmDA3PCu2vdKzWWg5579%2Forig&rpt=imageview&source-serpid=RELH2kmG1ZnBn7OUv1qEjw&source-serpid=ilqPo6CLfCRg4AA6IGFMCw&cbir_id=2271911%2FqGMDLJbmDA3PCu2vdKzWWg5579&cbir_page=similar&isize=large'

	browser.get(url)
	time.sleep(1)

	element = browser.find_element_by_tag_name("body")
	# Scroll down
	for i in range(count//10):
		element.send_keys(Keys.PAGE_DOWN)
		time.sleep(0.3)

	try:
		browser.find_element_by_id("smb").click()
		for i in range(count//10):
			element.send_keys(Keys.PAGE_DOWN)
			time.sleep(0.3)  # bot id protection
	except:
		for i in range((count//100)+1):
			element.send_keys(Keys.PAGE_DOWN)
			time.sleep(0.3)  # bot id protection

	print("Reached end of the page")
	time.sleep(0.5)

	source = browser.page_source
	soup = BeautifulSoup(source,'lxml') # choose lxml parser
	# find the tag : <img ... >
	image_tags = soup.findAll('img')
	# print out image urls
	counter = 0
	for image_tag in image_tags:
		
		
		if counter == count:
			break
		
		link = image_tag.get('src')
		
		if link is None or link == '':
			continue
			
		link = 'http:' + link
		print(link, counter+1) #page source
		
		for i in range(3):
			try:
				urllib.request.urlretrieve(link, "./"+query+"/" + query+ str(counter+1) + ".jpg")
				counter += 1
				break
			except:
				print("\tSleeping..")
				time.sleep(2)
		time.sleep(0.1)
	





# imgCount = int(input('Enter Image Count => \t'))
imgCount = int(args.count)
# queries = str(input('Enter Search Query with Commas => \t'))
queries = str(args.query)
query_list = queries.split(',')

for query in query_list:
	run(query, imgCount)