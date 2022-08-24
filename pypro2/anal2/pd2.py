# 연산
from pandas import Series, DataFrame
import numpy as np

print('------Series 객체 간 연산------')
s1 = Series([1,2,3], index=['a','b','c'])
s2 = Series([4,5,6,7], index=['a','b','d','c'])
print(s1)
print(s2)
print(s1+s2)  # Series 객체 간 연산.+,-,*,/(index가 같은 값을 연산함)
print(s1.add(s2))  # numpy의 명령을 계승받음

print()
print('------DataFrame 객체 간 연산------')
df1 = DataFrame(np.arange(9.).reshape(3,3), 
                columns=list('kbs'), index=['서울','대전','부산'])
df2 = DataFrame(np.arange(12.).reshape(4,3), 
                columns=list('kbs'), index=['서울','대전','제주','수원'])
print(df1)
print(df2)
print(df1 + df2)  # 대응되는 값이 없으면 NaN(결측치)
print(df1.add(df2, fill_value = 0))  # NaN을 0으로 채운 후에 연산에 참여한다.
# add, sub, mul, div

print()
seri = df1.iloc[0]
print(seri)
print(df1)
print()
print(df1 + seri)  # DataFrame/Series 연산 : Broadcasting

# 기술 통계 관련 함수 :  수집한 데이터를 요약, 묘사, 설명하는 통계 기법
print('------결측값 처리------')
df = DataFrame([[1.4, np.nan],[7, -1.5],[np.NaN, np.NAN],[0.5, -1]], columns=['one','two'])
print(df)
print()
print(df.drop(1))   # 1행(특정행) 삭제
print(df.isnull())  # null값 탐지
print(df.notnull()) # null 아닌 값 탐지
print(df.dropna())  # NaN이 하나라도 있으면 해당 행 삭제
print(df.dropna(how='any'))  # NaN이 하나라도 있으면 해당 행 삭제
print(df.dropna(how='all'))  # 모든 값이 NaN인 경우 해당 행 삭제
print(df.dropna(subset=['one']))  # 'one'칼럼값이 NaN이 있는 경우 해당행 삭제
print(df.dropna(axis='rows'))  # NaN이 있는 행의 해당행 삭제
print(df.dropna(axis='columns'))  # NaN이 있는 열의 해당행 삭제
print()
print(df.fillna(0))  # NaN값을 0으로 채움
print(df.fillna(method='ffill'))  # NaN값을 앞의 값으로 채움. 앞의 값이 없으면 그대로 NaN 출력 
print(df.fillna(method='bfill'))  # NaN값을 뒤의 값으로 채움

print()
print('--------내장 함수--------')
print(df)
print(df.sum())  # 열의 합
print(df.sum(axis=0))
print()
print(df.sum(axis=1))  # 행의 합
print(df.mean(axis=1))  # 행의 평균
print(df.mean(axis=1, skipna=False))  # 행의 평균. NaN은 연산에서 제외
print()
print(df.describe())  # 요약 통계량 출력
print(df.info())  # 구조 출력
print()
words = Series(['봄', '여름', '봄'])
print(words.describe())













