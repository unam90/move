# BeautifulSoup의 find / select 함수 연습

from bs4 import BeautifulSoup

html_page = """
<html><body>
<h1>제목</h1>
<p>웹 문서 읽기</p>
<p>원하는 자료 선택</p>
</body></html>
"""
print(html_page, type(html_page))  # <class 'str'>

soup = BeautifulSoup(html_page, 'html.parser')  # BeautifulSoup 객체 생성
print(soup, type(soup))  # <class 'bs4.BeautifulSoup'>

print()
h1 = soup.html.body.h1
print('h1: ', h1.string)  # h1.text라고 해도됨
p1 = soup.html.body.p
print('p1: ', p1.string)
p2 = p1.next_sibling.next_sibling
print('p2: ', p2.string)

print('find 함수 사용 : 반환값이 1개 ---------------')
html_page2 = """
<html><body>
<h1 id="title">제목 태그</h1>
<p>웹 문서 읽기</p>
<p id="my" class="our" kbs='공영방송'>원하는 자료 선택</p>
</body></html>
"""
print()
soup2 = BeautifulSoup(html_page2, 'lxml')
print(soup2.p, soup.p.string)  # <p>웹 문서 읽기</p> 웹 문서 읽기
print()
print(soup2.find('p').string)  # 웹 문서 읽기
print()
print(soup2.find(['p', 'h1']))  # <h1 id="title">제목 태그</h1>
print()
print(soup2.find('p', id='my').string)  # 원하는 자료 선택
print(soup2.find(id="my").string)  # 원하는 자료 선택
print()
print(soup2.find(id='title').string)  # 제목 태그
print()
print(soup2.find(class_="our").string)  # 원하는 자료 선택
print(soup2.find(attrs={'class':'our'}).string)  # 원하는 자료 선택
print(soup2.find(attrs={'id':'my'}).text)  # 원하는 자료 선택
print(soup2.find(attrs={'kbs':'공영방송'}).text)  # 원하는 자료 선택
print()

print('find_all, findAll 함수 사용 : 반환값이 복수---------------')
html_page3 = """
<html><body>
<h1 id="title">제목 태그</h1>
<p>웹 문서 읽기</p>
<p id="my" class="our" kbs='공영방송'>원하는 자료 선택</p>
<div>
    <a href="https://www.naver.com">네이버</a>
    <a href="https://www.daum.net">다음</a>
</div>
</body></html>
"""
soup3 = BeautifulSoup(html_page3, 'lxml')
print(soup3.find_all('a'))  # 모든 a태그를 반환함
print(soup3.find_all(['p']))  # 모든 p태그를 반환함
print(soup3.findAll(['p', 'a']))  # 모든 p태그, a태그를 반환함 / find_all 과 findAll은 같다.
print()
links = soup3.find_all('a')
for i in links:
    href = i.attrs['href']  # href의 속성값을 가져옴
    text = i.string
    print(href,' ', text)

print('-------------------')
# https://music.bugs.co.kr/chart 에서 자료 읽기
from urllib.request import urlopen
url = urlopen("https://music.bugs.co.kr/chart")
soup = BeautifulSoup(url.read(), 'html.parser')
# print(soup)
musics = soup.findAll('td', class_="check")
#print(musics)
print()

for i, music in enumerate(musics):
    print('{}위 : {}'.format(i + 1, music.input['title'])) # music의 input속성에 title

print()
print('select_one, select 함수 사용 : 반환값이 복수---------------')
html_page4 = """
<html><body>
<div id='hello'>
    <a href="https://www.naver.com">네이버</a>
    <span>
        <a href="https://www.daum.net">다음</a>
    </span>
    <ul class="world">
        <li>여름은</li>
        <li>즐거워</li>
    </ul>
</div>

<div id='hi' class='good'>
두번째 div 태그
</div>
</body></html>
"""
soup4 = BeautifulSoup(html_page4, 'lxml')
print(soup4.select_one("div"))
print()
print(soup4.select_one("div a"))
print(soup4.select_one("div > a"))  
print()
print(soup4.select("div a"))    # div 태그의 자손
print(soup4.select("div > a"))  # div 태그의 직계 자식
print()
print(soup4.select("div#hello")) # div 태그 중 아이디가 hello
print(soup4.select("div#hi"))    # div 태그 중 아이디가 hi
print(soup4.select("div.good"))  # div 태그 중 클래스가 good
print()
print(soup4.select("div#hello ul.world > li"))  # div 태그 중 아이디가 hello 자손으로 ul 중 class가 world인 직계 자식 li 
print()
imsi = soup4.select("div#hello ul.world > li")
for i in imsi:
    print(i.string)








