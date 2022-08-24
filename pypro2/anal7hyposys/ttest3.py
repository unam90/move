# 어느 한 음식점을 대상으로 비(눈)가 올 때의 매출 / 비(눈)가 오지 않을 때의 매출의 평균의 차이 검정

# 귀무 : 강수 여부에 따른 매출액에 차이가 없다.
# 대립 : 강수 여부에 따른 매출액에 차이가 있다.

import numpy as np
import scipy.stats as stats
import pandas as pd

# 매출 데이터 읽기
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tsales.csv',
                         dtype={'YMD':'object'})  # int타입을 object으로 변환해서 읽어옴(병합을 위해)
    
print(sales_data.head(5))
print(sales_data.info())
print()

# 날씨 데이터 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tweather.csv')
print(wt_data.head(3))
# print(wt_data.info())

print()
# wt_data의 날짜를 2018-06-01 ==> 20180601로 변환(병합을 위함)
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-', ''))
print(wt_data.head(3))
print()
frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm')
print(frame.head(3))
print(frame.columns)  # column확인
# 'YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes'

data = frame.iloc[:, [0,1,7,8]]  # 'YMD', 'AMT', 'maxTa', 'sumRn' 열을 가져오기
print(data.head(3))

# 결측치 확인
print(data.isnull().sum())  # 결측치 없음

print('독립표본 t검정-------------')
# print(data['sumRn'] > 0)  # 비가 왔는지 아닌지 확인 True/False

# data['rain_yn'] = (data['sumRn'] > 0).astype(int)  # 비가 왔으면:1, 안왔으면:0
# print(data.head(3))

print(True * 1, False * 1)
data['rain_yn'] = (data['sumRn'] > 0) * 1
print(data.head(3))

# box plot으로 시각화
sp = np.array(data.iloc[:, [1, 4]])  # AMT, rain_yn 추출
# print(sp)  # [[      0       0][  18000       1][  50000       0]... 2차원 배열(매출액과 강수여부)
tg1 = sp[sp[:, 1]==0, 0]  # 집단1 : 비 안올 때 매출액
tg2 = sp[sp[:, 1]==1, 0]  # 집단2 : 비 올 때 매출액
print(tg1[:3])
print(tg2[:3])

import matplotlib.pyplot as plt
# plt.plot(tg1)
# plt.show()
# plt.plot(tg2)
# plt.show()
plt.boxplot([tg1, tg2])
plt.show()
print(np.mean(tg1), ' ', np.mean(tg2)) # 761040.2542372881   757331.5217391305

# 정규성 확인
print(stats.shapiro(tg1).pvalue)  # 0.056049 > 0.05 만족
print(stats.shapiro(tg2).pvalue)  # 0.882739 > 0.05 만족

# 등분산성
print(stats.levene(tg1, tg2).pvalue)  # 0.712345 > 0.05 만족

print(stats.ttest_ind(tg1, tg2, equal_var=True))
# Ttest_indResult(statistic=0.10109828602924716, pvalue=0.919534587722196)
# 해석 : pvalue=0.9195 > 0.05 이므로 귀무가설 채택
# 강수 여부에 따른 매출액의 평균은 영향이 없는 것으로 판정.
