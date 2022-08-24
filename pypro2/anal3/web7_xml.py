# XML 형식 문서 읽기 : BeautifulSoup
import urllib.request as req
from bs4 import BeautifulSoup

url = "https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.xml"
plainText = req.urlopen(url).read().decode()
# print(plainText)

xmlObj = BeautifulSoup(plainText, 'lxml')
libData = xmlObj.select('row')
# print(libData)

for data in libData:
    name = data.find('lbrry_name').text 
    addr = data.find('adres').string 
    print('도서관명 :', name, end=' ')
    print('주소 :', addr)