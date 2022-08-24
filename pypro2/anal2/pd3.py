    import pandas as pd
import numpy as np

# dataFrame 구조 변경 : stack, unstack

df = pd.DataFrame(np.arange(6).reshape(2, 3),
                  index=['대전', '서울'], columns=['2020', '2021', '2022'])
print(df)
df_row = df.stack()  # 열을 행으로 변경 : index를 기준으로 열쌓기
print(df_row)
print(df)
df_col = df_row.unstack()  # stack 결과를 해제
print(df_col)

print('------연속형 자료 범주화 (구간 설정)-----')
price = [10.3, 5.5, 7.8, 3.6]
cut = [3, 7, 9, 11]  # 구간 기준값
result_cut = pd.cut(price, cut)  # price값을 cut를 기준으로 범주화
print(result_cut)  # (a, b]  <== a초과 b이하 라는 뜻
print(pd.value_counts(result_cut))  # 3초과 7 이하는 2개, 7초과 9이하는 1개, 9초과 11이하는 1개
print(type(result_cut))  # <class 'pandas.core.arrays.categorical.Categorical'>
print()
datas = pd.Series(np.arange(1, 1001))
print(datas.head(3))
print(datas.tail(3))
print()
result_cut2 = pd.qcut(datas, 3)  # target에 대해 구간수를 직접 지정
print(result_cut2)
print(pd.value_counts(result_cut2))

print()
group_col = datas.groupby(result_cut2)
print(group_col.agg(['count', 'mean', 'std', 'min']))  # 그룹에 대한 함수를 실행(agg)
print()
# agg 대신 함수를 직접 작성하기 
def summary_func(gr):
    return {'count':gr.count(), 'mean':gr.mean(), 'std':gr.std(), 'min':gr.min()}

print(group_col.apply(summary_func))
print()
print(group_col.apply(summary_func).unstack())



