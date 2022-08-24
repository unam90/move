# pandas : 고수준의 자료구조와 빠르고 쉬운 분석용 자료구조 및 함수를 지원
# pandas로 data munging or data wrangling 작업을 효율적으로 처리가 가능하다.
# 원자료(raw data)를 보다 쉽게 접근하고 분석할 수 있도록 데이터를 정리하고 통합하는 과정

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# Series : 일련의 자료를 담을 수 있는 1차원 배열과 유사한 자료구조로 색인을 갖음

obj = pd.Series([3, 7, -5, 4])  # list 
# obj = pd.Series((3, 7, -5, 4))  # tuple
# obj = pd.Series({3, 7, -5, 4})  # set은 순서가 없기 때문에 인덱싱이 되지 않음. TypeError: 'set' type is unordered
print(obj, type(obj))  # list와 tuple은 자동 인덱싱 됨

obj2 = pd.Series([3, 7, -5, 4], index=['a','b','c','d'])  
print(obj2)
print(obj2.sum(), sum(obj2), np.sum(obj2))
print(obj2.mean(), obj2.std())  # 평균, 표준편차

print()
print(obj2.values)  # 콤마가 없으면 ndarray, 콤마가 있으면 list
print(obj2.index)
print()
print('--------슬라이싱------------')
print(obj2['a'], ' ', obj2[['a']])  # 3   a    3
print(obj2[0])
print(obj2[['a','b']])  # 색인명을 써서 슬라이싱
print(obj2['a':'c'])
print(obj2[1:4])  # 1번이상 4번미만까지 슬라이싱
print(obj2[[2,1]])  # 2번째와 1번째 
print(obj2 > 1)  # 1보다 큰 값은 True, 아니면 False
print('a' in obj2)

print('-----dict type으로 Series 객체 생성-----')
names = {'mouse':5000, 'keyboard':35000, 'monitor':550000}
obj3 = Series(names)
print(obj3, type(obj3))  
obj3.index = ['마우스', '키보드', '모니터']
print(obj3)
print(obj3['마우스'])
print(obj3[0])

obj3.name = '상품가격'
print(obj3)

print()
print('-----DataFrame : 표 모양의 자료구조-----------------')
df = DataFrame(obj3)  # Series로 DataFrame을 만듬
print(df, type(df))

data = {
'irum':['홍길동', '한국인', '신기해', '공기밥', '한가해'],
'juso':('역삼동', '신당동', '역삼동', '역삼동', '신사동'),
'nai':[23, 25, 33, 30, 35],
}
print(data, type(data))  # dict type

frame = DataFrame(data)  # dict타입으로 DataFrame을 만듬
print(frame)
print(frame.irum, type(frame.irum))
print(frame['irum'])

# Series와 DataFrame은 자동으로 인덱스가 붙어있다.

print(DataFrame(data, columns=['juso', 'nai', 'irum']))

print()
frame2 = DataFrame(data, columns=['irum','juso','nai','tel'], 
                   index=['a','b','c','d','e'])
print(frame2)  # tel 자리에 NaN (결측값)

frame2['tel'] = '111-1111'
print(frame2)

val = Series(['222-1111', '333-1111', '444-1111'], index=['b','c','e'])
print(val)

frame2['tel'] = val  # 덮어쓰기
print(frame2)

print()
print(frame2.T)  # 행렬이 위치 변경(Transpose)

print(frame2.values)  # 값들을 matrix로 돌려줌
print(frame2.values[0,1])  # 0행 1열의 값을 돌려줌  
print(frame2.values[0:2])  # 0행에서 1행까지의 값
print(type(frame2.values[0:2]))  # <class 'numpy.ndarray'>

print()
print('-----행 또는 열 삭제-----')  # axis=0 행, axis=1 열
frame3 = frame2.drop('d')  # axis=0 을 생략한 것. d행 삭제
print(frame3) 
print()
frame4 = frame2.drop('tel', axis=1)  # tel 열 삭제
print(frame4) 

print()
print('--------정렬--------')
print(frame2.sort_index(axis=0, ascending=False))  # 행단위 정렬 descending
print(frame2.sort_index(axis=1, ascending=False))  # 열단위 정렬 descending
print(frame2.rank(axis=0))  # 순위가 매겨짐

counts = frame2['juso'].value_counts()
print('칼럼값 갯수 : ', counts)

print()
print('-----문자열 자르기-----')
data = {
    'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon':[23, 25, 15]
}
fr = DataFrame(data)
print(fr)
print()
result1 = Series([x.split()[0] for x in fr.juso])
result2 = Series((x.split()[1] for x in fr.juso))
print(result1)
print(result2)
print(result1.value_counts())

print()
print('-------Series의 재색인-------')
data = Series([1,3,2], index=(1,4,2))  # index는 unique하기 때문에 set도 가능
print(data)
data2 = data.reindex((1,2,4))  # reindex로 행의 순서를 바꿀 수 있음
print(data2)
print()
print('-----재색인 시 값 채워넣기-----')
data3 = data2.reindex([0, 1, 2, 3, 4, 5])  # 대응값이 없는 index는 NaN(결측치)가 된다.
print(data3)

data3 = data2.reindex([0,1,2,3,4,5], fill_value = 77)  # 대응값이 없을 시 77로 채움
print(data3)

# data3 = data2.reindex([0,1,2,3,4,5], method = 'ffill')  # 대응값이 없을 시 앞의 값으로 채우기(forward fill) 
data3 = data2.reindex([0,1,2,3,4,5], method = 'pad')  # 대응값이 없을 시 앞의 값으로 채우기
print(data3)

# data3 = data2.reindex([0,1,2,3,4,5], method = 'bfill')  # 대응값이 없을 시 뒤의 값으로 채우기(back fill)
data3 = data2.reindex([0,1,2,3,4,5], method = 'backfill')  # 대응값이 없을 시 뒤의 값으로 채우기(back fill)
print(data3)

print()
print('------bool 처리------')
df = DataFrame(np.arange(12).reshape(4, 3),
               index = ['1월','2월','3월','4월'], columns=['강남','강북','서초'])
print(df)
print()
print(df['강남'] > 3)  # 3번째 행 이상 출력
print(df[df['강남'] > 3])  # 조건이 참인 행 출력

print()
print(df <3)  # 3보다 작으면 True, 아니면 False
print()
df[df < 3] = 0  # 3보다 작은 값은 0으로 대체
print(df)

print()
print('DataFrame 관련 슬라이싱 함수 : loc() - 라벨 지원, iloc() - 순서 지원') 
print(df.loc['3월', :])  # 3월행 모든열 출력
print(df.loc['3월', ])   # 3월행 모든열 출력
print(df.loc[:'2월'])    # 2월행 이하 출력
print(df.loc[:'2월',['서초']])  # 2월행 이하, 서초열 출력
print()
print(df.iloc[2])     # 2행 모두 출력
print(df.iloc[2,])    # 2행 모두 출력
print(df.iloc[2, :])  # 2행 모두 출력
print()
print(df.iloc[:3])    # 3행 미만 출력
print(df.iloc[:3, 2]) # 3행 미만 2열 출력
print(df.iloc[1:3, 1:3]) # 1행~2행, 1열~2열 출력















