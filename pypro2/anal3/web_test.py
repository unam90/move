# 상품명과 가격 출력 , 가격의 평균, 표준편차 출력
import urllib.request as req
from bs4 import BeautifulSoup

url = "https://kyochon.com/menu/chicken.asp"
chicken = req.urlopen(url)
soup = BeautifulSoup(chicken, 'html.parser')
# print(soup)
#tabCont01 > ul > li:nth-child(1) > a > dl > dt
print(soup.select_one("div#tabCont01 > ul > li:nth-child(1) > a > dl > dt").string)
# print()
menus = soup.select("dl > dt")
# print(datas)
menuDatas=[]
for m in menus:
    # print(m.string)
    menuDatas.append(m.string)
    
#tabCont01 > ul > li:nth-child(1) > a > p.money > strong
# print(soup.select_one("div#tabCont01 > ul > li:nth-child(1) > a > p.money > strong").string)
prices = soup.select("p.money > strong")
# print(prices)
priceDatas =[]
for p in prices:
    print(p.string)

