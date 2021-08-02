import os
import re
import time
import socket

from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    ElementNotInteractableException,
)

from PIL import Image
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from selenium.webdriver.chrome.options import Options


class Crawler:
    def __init__(self):
        # 이미지들이 저장될 경로 및 폴더 이름
        self.path = "H:\\Resources\\pytorch_dataset\\cars"
        self.date = str(date.today())

        # 검색어 입력 및 중복 검사
        self.query = input("입력: ")
        while self.checking(self.query) is True:
            self.query = input("입력: ")

        # 웹 브라우저의 개발자 모드(F12)를 열어 console에 navigator.userAgent라고 입력 후 출력되는 값을 복사
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        opts = Options()
        opts.add_argument(f"user-agent={user_agent}")

        # 드라이버 생성
        self.driver = webdriver.Chrome("H:\\chromedriver", options=opts)

        # clickAndRetrieve() 과정에서 urlretrieve이 너무 오래 걸릴 경우를 대비해 타임 아웃 지정
        socket.setdefaulttimeout(30)

        # 크롤링한 이미지 수
        self.crawled_count = 0

    def scroll_down(self):
        scroll_count = 0

        print("> 스크롤 다운 시작")

        # 스크롤 위치값 얻고 last_height 에 저장
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        # 결과 더보기 버튼을 클릭했는지 유무
        after_click = False

        while True:
            print(f"> 스크롤 횟수: {scroll_count}")
            # 스크롤 다운
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            scroll_count += 1
            time.sleep(2)

            # 스크롤 위치값 얻고 new_height 에 저장
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            # 스크롤이 최하단이며
            if last_height == new_height:

                # 결과 더보기 버튼을 클릭한적이 있는 경우
                if after_click is True:
                    print("> 스크롤 다운 종료")
                    break

                # 결과 더보기 버튼을 클릭한적이 없는 경우
                elif after_click is False:
                    if self.driver.find_element_by_xpath(
                        '/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div[1]/div[2]/div[2]/input'
                    ).is_displayed():
                        self.driver.find_element_by_xpath(
                            '/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div[1]/div[2]/div[2]/input'
                        ).click()
                        print("> 결과 더보기 클릭")
                        after_click = True
                    elif NoSuchElementException:
                        print("> 이미지 갯수가 100개 이하임")
                        print("> 스크롤 다운 종료")
                        break

            last_height = new_height

    def click_img(self, img):
        # 이미지 클릭
        # img.click()
        self.driver.execute_script("arguments[0].click();", img)
        time.sleep(1.5)

        # 이미지 주소 반환
        image_src = '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img'

        src = self.driver.find_element_by_xpath(image_src).get_attribute("src")

        return src

    def retrieve(self, src):
        try:
            # 확장자
            if re.search(r"jpeg|png", src):
                ext = re.search(r"jpeg|png", src).group()
            else:
                ext = "jpg"

            # 저장될 이미지 파일 경로
            dst = f"{self.path}/{self.date}/{self.query}/{self.crawled_count + 1}.{ext}"
            self.crawled_count += 1

            # 이미지 저장
            urlretrieve(src, f"{dst}")
            print(f"> {self.crawled_count} / {self.len_list} 번째 사진 저장 ({dst})")

        except HTTPError:
            print("> HTTPError & 패스")
            pass

    def get_img_list(self):
        print("> 크롤링 시작")

        # 이미지 고급검색 중 이미지 유형 '사진'
        url = f"https://www.google.com/search?as_st=y&tbm=isch&hl=ko&as_q={self.query}&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=itp:photo"
        self.driver.get(url)
        self.driver.maximize_window()
        self.scroll_down()

        div = self.driver.find_element_by_xpath('//*[@id="islrg"]/div[1]')
        # class_name에 공백이 있는 경우 여러 클래스가 있는 것이므로 아래와 같이 css_selector로 찾음
        img_list = div.find_elements_by_css_selector(".rg_i.Q4LuWd")

        self.len_list = len(img_list)

        return img_list

    def get_src_list(self, img_list):
        src_list = []
        for idx, img in enumerate(img_list):
            try:
                src = self.click_img(img)
                print(f"{idx + 1} 번째 이미지 주소: {src}")
                src_list.append(src)

            except ElementClickInterceptedException:
                print("> ElementClickInterceptedException")
                self.driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
                print("> 100만큼 스크롤 다운 및 3초 슬립")
                time.sleep(3)
                # src = self.click_img(idx+1, img)

            except NoSuchElementException:
                print("> NoSuchElementException")
                self.driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
                print("> 100만큼 스크롤 다운 및 3초 슬립")
                time.sleep(3)
                # src = self.click_img(idx+1, img)

            except ConnectionResetError:
                print("> ConnectionResetError & 패스")

            except URLError:
                print("> URLError & 패스")

            except socket.timeout:
                print("> socket.timeout & 패스")

            except socket.gaierror:
                print("> socket.gaierror & 패스")

            except ElementNotInteractableException:
                print("> ElementNotInteractableException")

        self.driver.quit()

        # 다운로드 디렉토리 생성
        os.makedirs(self.path + "/" + self.date + "/" + self.query)
        print(f"> {self.path}/{self.date}/{self.query} 생성")

        return src_list

    def end_crawling(self):
        try:
            print(
                "> 크롤링 종료 (성공률: %.2f%%)" % (self.crawled_count / self.len_list * 100.0)
            )

        except ZeroDivisionError:
            print("> img_list 가 비어있음")

    def filtering(self, size):
        print("> 필터링 시작")

        filtered_count = 0
        dir_name = f"{self.path}/{self.date}/{self.query}"
        for index, file_name in enumerate(os.listdir(dir_name)):
            try:
                file_path = os.path.join(dir_name, file_name)
                img = Image.open(file_path)

                # 이미지 해상도의 가로와 세로 중 하나라도 size 이하인 경우
                if img.width <= size or img.height <= size:
                    img.close()
                    os.remove(file_path)
                    print(f"> {index} 번째 사진 삭제")
                    filtered_count += 1

            # 이미지 파일이 깨져있는 경우
            except OSError:
                os.remove(file_path)
                filtered_count += 1

        print(f"> 필터링 종료 (총 갯수: {self.crawled_count - filtered_count})")

    def checking(self, query):
        # 입력 받은 검색어가 이름인 폴더가 존재하면 중복으로 판단
        for dir_name in os.listdir(self.path):
            file_list = os.listdir(f"{self.path}/{dir_name}")
            if query in file_list:
                print(f"> 중복된 검색어: ({dir_name})")
                return True


if __name__ == "__main__":
    crawler = Crawler()
    img_list = crawler.get_img_list()
    src_list = crawler.get_src_list(img_list)

    # max_workers = 사용하려는 쓰레드 수
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_list = executor.map(crawler.retrieve, src_list)

    crawler.end_crawling()
    # crawler.filtering(540)      # 최소 960x540
