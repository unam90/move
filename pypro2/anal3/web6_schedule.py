# 일정 시간 마다 웬 문서 정기적으로 읽기

import datetime
import time
import urllib.request as req
from bs4 import BeautifulSoup

def working():
    url = "https://finance.naver.com/marketindex/"
    data = req.urlopen(url)
    
    soup = BeautifulSoup(data, 'lxml')
    # print(soup)  #exchangeList > li.on > a.head.usd > div > span.value
    price = soup.select_one("div.head_info > span.value").string
    print('미국 USD : ', price)
    
    # price 값을 파일로 저장하기 (년-월-일-시-분-초.txt)
    t = datetime.datetime.now()
    # print(t)
    fname = "./usd/" + t.strftime('%Y-%m-%d-%H-%M-%S') + '.txt'
    # print(fname)  # ./usd/2022-07-20-14-48-55.txt
    
    with open(fname, mode='w') as f:
        f.write(price)

while True:
    working()
    time.sleep(3)
