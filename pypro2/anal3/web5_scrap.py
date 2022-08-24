# 웹 문서 읽기
import urllib.request as req
from bs4 import BeautifulSoup
import urllib

# 1) 위키백과에서 이순신으로 검색된 자료 읽기
url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
wiki = req.urlopen(url)

soup = BeautifulSoup(wiki, 'html.parser')
# #mw-content-text > div.mw-parser-output > p:nth-child(6)  / copy selector
print(soup.select('#mw-content-text > div.mw-parser-output > p'))
print()
result = soup.select('div.mw-parser-output > p > b')
print(result)

for r in result:
    print(r.string)
    
print()
print('-----다음 사이트의 뉴스 사회면 문서 읽기-------')
url = "https://news.daum.net/society#1"
daum = req.urlopen(url)
soup = BeautifulSoup(daum, 'html.parser')
# body > div.direct-link
print(soup.select_one("body > div.direct-link > a").string)
print()
datas = soup.select("div.direct-link > a")

for i in datas:
    href=i.attrs['href']
    text=i.string
    print('href : %s, text : %s'%(href, text))

print()
print('-- a 태그 읽기 ---')
datas = soup.findAll('a')   #상위 5개 a 태그만 읽어오기
for i in datas[:5]:
    href=i.attrs['href']
    text=i.string
    print('href : %s, text : %s'%(href, text))