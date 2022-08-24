# 웹 문서를 읽어 형태소 분석 : 명사만 추출 후 ...
import urllib
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from urllib import parse

okt = Okt()

# 위키백과 이순신 등의 단어 관련 문서 읽기
para = parse.quote('이순신')  # 한글 인코딩
url = 'https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0'
page = urllib.request.urlopen(url)
soup = BeautifulSoup(page.read(), 'lxml')
# print(soup)

wordlist = []

for item in soup.select('#mw-content-text > div > p'):
    if item.string != None:
        # print(item.string)
        wordlist += okt.nouns(item.string)
        
print('wordlist:', wordlist)
print('단어 수:', len(wordlist))

print()
word_dict = {}

for i in wordlist:
    if i in word_dict:
        word_dict[i] += 1
    else:
        word_dict[i] = 1
        
print('word_dict:', word_dict)

setdata = set(wordlist)
print(setdata)
print('발견된 단어 수:', len(setdata))

print()
import pandas as pd
df1 = pd.DataFrame(wordlist, columns=['단어'])
print(df1.head(3))
print()
df2 = pd.DataFrame(word_dict.keys(), word_dict.values())
print(df2)
df2 = df2.T 
df2.columns = ['단어', '빈도수']
print(df2.head(3))

df2.to_csv('이순신.csv', sep=',', index = False)

print()
df3 = pd.read_csv('이순신.csv')
print(df3.head(5))



    
    
    
    
    