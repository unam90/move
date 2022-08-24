import pandas as pd
import numpy as np

# DataFrame 객체 병합 : merge 
df1 = pd.DataFrame({'data1': range(7), 'key':['b','b','a','c','a','a','b']}) 
print(df1)
df2 = pd.DataFrame({'key':['a','b','d'], 'data2': range(3)}) 
print(df2)
print('------ inner join ------')
print(pd.merge(df1, df2, on='key'))  # 'key' 를 기준으로 병합(inner join : 교집합)
print()
print(pd.merge(df1, df2, on='key', how='inner'))  # how(병합방법)은 inner
print()
print('------ outer join ------')
print(pd.merge(df1, df2, on='key', how='outer'))  # key를 기준으로 병합 (full outer join)
print()
print('---- left outer join ----')
print(pd.merge(df1, df2, on='key', how='left'))  # key를 기준으로 병합 (left outer join)
print()
print('---- right outer join ----')
print(pd.merge(df1, df2, on='key', how='right'))  # key를 기준으로 병합 (right outer join)

print()
print('----공통 칼럼명이 없는 경우(df1 vs df3)----')
df3 = pd.DataFrame({'key2':['a','b','d'], 'data2': range(3)})
print(df3)
print(df1)
print(pd.merge(df1, df3, left_on='key', right_on='key2', how='inner')) 
print()
print('------ 자료 이어 붙이기 ------')
print(pd.concat([df1, df3], axis=0))  # 행단위로 이어붙이기
print()
print(pd.concat([df1, df3], axis=1))  # 열단위로 이어붙이기

print()
print('----- 피봇(pivot) -----')
# 열을 기준으로 구조를 변경하여 새로운 집계표를 작성
data = {'city':['강남', '강북', '강남', '강북'],
        'year':[2000, 2001, 2002, 2002],
        'pop':[3.3, 2.5, 3.0, 2.0]}

df = pd.DataFrame(data)
print(df)
print()
print('---- pivot ----')
print(df.pivot('city', 'year', 'pop'))
print()
# set_index : 기존의 행 인덱스를 제거하고 첫번째열 인덱스를 설정
print(df.set_index(['city', 'year']).unstack())

print()
print('---- group by ----')
hap = df.groupby(['city'])
print(hap.sum())
print(df.groupby(['city']).sum())  # 위 두줄을 한 줄로 표현

print(df.groupby(['city', 'year']).mean())  # city별 year별 평균을 구함
print()
print(df.groupby(['city']).agg('sum'))  # city별 합
print()
print(df.groupby(['city', 'year']).agg('sum'))  
print(df.groupby(['city', 'year']).agg(['mean','sum'])) # city별 year별 평균과 합을 구함

print()
print('---- pivot_table ----')
print(df)
print(df.pivot_table(index=['city']))  # 평균을 기본으로 계산
print(df.pivot_table(index=['city'], aggfunc=np.mean))  # 기본값 (위와 동일)
print(df.pivot_table(index=['city', 'year'], aggfunc=[len, np.sum]))
print(df.pivot_table(values=['pop'], index=['city']))  # city별 pop의 평균
print(df.pivot_table(values=['pop'], index=['city'], aggfunc=np.mean))  # city별 pop의 평균
print(df.pivot_table(values=['pop'], index=['city'], aggfunc=len))
print(df.pivot_table(values=['pop'], index=['year'], columns=['city']))
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], 
                     margins=True))
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], 
                     margins=True, fill_value=0))

























