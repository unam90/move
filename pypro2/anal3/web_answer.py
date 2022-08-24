import urllib.request as req
from bs4 import BeautifulSoup

url = "https://kyochon.com/menu/chicken.asp"
data = req.urlopen(url)

soup = BeautifulSoup(data, 'lxml')

name = list()
for tag in soup.select('dl.txt > dt'):
    name.append(tag.text.strip())
    
price= list()
for tag in soup.select('p.money strong'):
    temp = tag.text.strip().replace(',','')
    price.append(int(temp))
    
import pandas as pd
df = pd.DataFrame(name, columns=['상품명'])
df['가격'] = price
print(df)

print('가격 평균 : ', round(df.가격.mean(), 2))
print('가격 표준편차 : ', round(df.가격.std(), 2))
