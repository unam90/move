# 웹 검색 문서로 형태소 분석 후 워드클라우드 만들기 
# pip install pygame
# pip install simplejson
# pip install pytagcloud

"""
import urllib.request
from bs4 import BeautifulSoup
from urllib.parse import quote

# keyword = input('검색어:')
keyword = '무더위'
print(quote(keyword))
url = 'https://www.donga.com/news/search?query=' + quote(keyword)
page = urllib.request.urlopen(url)
soup = BeautifulSoup(page, 'lxml')
print(soup)

msg = ''
for title in soup.find_all('p', 'tit'):
    title_link = title.select('a')
    # print(title_link)
    article_url = title_link[0]['href']
    # print(article_url)
    try:
        sou_article = urllib.request.urlopen(article_url)
        soup = BeautifulSoup(sou_article, 'lxml')
        contents = soup.select('div.article_txt')
        # print(contents)
        for imsi in contents:
            item = str(imsi.find_all(text=True))
            # print(item)
            msg = msg + item
        
    except Exception as e:
        pass
print(msg)

from konlpy.tag import Okt
from collections import Counter  # 단어 수를 세어주는 라이브러리 

okt = Okt()
nouns = okt.nouns(msg)

result = []
for imsi in nouns:
    if len(imsi) > 1:
        result.append(imsi)

print(result)
count = Counter(result)
print(count)
tag = count.most_common(50)  # 상위 50개만 워드클라우드에 참여 

import pytagcloud
taglist = pytagcloud.make_tags(tag, maxsize=100)
print(taglist)

pytagcloud.create_tag_image(taglist, output='word.png', size=(1000,600), 
                            background=(0,0,0), fontname='korean', rectangular=False)
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('word.png')
plt.imshow(img)
plt.show()

    
    
    
    



