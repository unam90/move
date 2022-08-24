# * Pandas의 DataFrame 관련 연습문제 *
#   pandas 문제 1)
#   a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오. 
#      np.random.randn(9, 4)
#   b) a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오
#   c) 각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

print('a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오. ')
np.random.seed(123)
df = pd.DataFrame(np.random.randn(9, 4))
print(df)

print()
print('b) a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오.')
df.columns = ['No1', 'No2', 'No3', 'No4']
print(df)

print()
print('c) 각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용')
print(df.mean(axis=0))

print()
# pandas 문제 2)
# a) DataFrame으로 위와 같은 자료를 만드시오. column(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.
# b) c row의 값을 가져오시오.
# c) a, d row들의 값을 가져오시오.
# d) numbers의 합을 구하시오.
# e) numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.

print()
print('a)column(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.')
df1 = pd.DataFrame([[i] for i in range(10,41,10)], 
                    columns=['numbers'], index=['a','b','c','d'])

print(df1)
print()
print('b) c row의 값을 가져오시오.')
print(df1.loc['c'])
print()
print('c) a, d row들의 값을 가져오시오.')
print(df1.loc[['a', 'd']])

print()
print('d) numbers의 합을 구하시오.')
print(df1.numbers.sum())

print()
print('e) numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.')
print(df1 ** 2)

print()


















