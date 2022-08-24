# 형태소 분석 : 자연어를 형태소를 비롯하여 어근, 접두사, 접미사, 품사 등 다양한 언어적 속성의 구조를 파악
# 영문 : nltk 
# 한글 : konlpy
from konlpy.tag import Kkma, Okt, Komoran

# kkma = Kkma()
# print(kkma.sentence('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
# print(kkma.nouns('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
# print(kkma.morphs('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))

print()
okt = Okt()
print(okt.nouns('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(okt.morphs('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(okt.pos('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(okt.pos('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다', stem=True))  # 원형 어근 출력
print(okt.phrases('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))

print()
ko = Komoran()
print(ko.nouns('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(ko.morphs('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))
print(ko.pos('한글 데이터 분석을 위한 라이브러리를 설치합니다. 행운을 빕니다'))