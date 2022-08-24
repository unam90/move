import pandas as pd
import numpy as np

# pandas로 파일 읽기
# df = pd.read_csv('testdata/ex1.csv')
# df = pd.read_csv('testdata/ex1.csv', sep=',')
df = pd.read_table('testdata/ex1.csv', sep=',')
print(df, type(df))

print()
df = pd.read_csv('testdata/ex2.csv', header=None, names=['a','b','c','d','f'])  # header가 없으면 칼럼명이 자동으로 붙음
print(df)

print()
df = pd.read_csv('testdata/ex3.txt')
print(df)

print()
df = pd.read_csv('testdata/ex3.txt', sep='\s+')  # 구분자로 정규표현식 \s+ 공백=1개 이상
print(df)

print()
df = pd.read_table('testdata/ex3.txt', sep='\s+', skiprows=[1,3])
print(df)

print()
mydf = pd.read_fwf('testdata/data_fwt.txt', widths=(10,3,5), 
                   header=None, names=('date', 'name', 'price'))
print()
print(mydf)
print()
print(mydf['date'])

print()
# 큰 규모의 파일인 경우 나누어 읽기 : chunk 
test = pd.read_csv('testdata/data_csv2.csv', header=None, chunksize=3)
print(test)  # TextFileReader object

for piece in test:
    # print(piece)
    print(piece.sort_values(by=2, ascending=True))  # 2번째 열 기준 오름차순 정렬

print()
print('-------------------------')
# pandas 객체를 파일로 저장
items = {'apple':{'count':10, 'price':1500}, 'orange':{'count':5, 'price':700}}
df = pd.DataFrame(items)
print(df)
print()
df.to_clipboard()
print(df.to_html())
print(df.to_csv())
print(df.to_json())

print()
df.to_csv('result1.csv', sep=',')
df.to_csv('result2.csv', sep=',', index=False)
df.to_csv('result3.csv', sep=',', index=False, header=False)











