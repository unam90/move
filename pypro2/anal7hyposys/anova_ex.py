# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.
# 귀무 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하지 않는다.
# 대립 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재한다.

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

data = pd.read_csv('anova_ex.txt', sep=' ')
print(data)
print(data.isnull().sum())
data = data.fillna(data.quantity.mean())
print(data)

q1 = data[data['kind']==1]
q2 = data[data['kind']==2]
q3 = data[data['kind']==3]
q4 = data[data['kind']==4]

print()
# 정규성 검정
print(stats.ks_2samp(q1.quantity, q2.quantity).pvalue)  # 0.2285 > 0.05 만족
print(stats.ks_2samp(q2.quantity, q3.quantity).pvalue)  # 0.8857 > 0.05 만족
print(stats.ks_2samp(q3.quantity, q4.quantity).pvalue)  # 0.6 > 0.05 만족
print(stats.ks_2samp(q1.quantity, q3.quantity).pvalue)  # 1.0 > 0.05 만족
print(stats.ks_2samp(q1.quantity, q4.quantity).pvalue)  # 0.6 > 0.05 만족
print(stats.ks_2samp(q2.quantity, q4.quantity).pvalue)  # 0.2285 > 0.05 만족

# print()
# # ANOVA 검정 
# import statsmodels.api as sm
# print('평균1 :', q1.mean())  # 평균1 : 74.9444
# print('평균2 :', q2.mean())  # 평균2 : 61.5
# print('평균3 :', q3.mean())  # 평균3 : 70.6666
# print('평균4 :', q4.mean())  # 평균4 : 75.0
#
# lm = ols("data['kind'] ~ data['quantity']", data=data).fit()
# result = sm.stats.anova_lm(lm, typ=2)
# print(result)
# # 해석 : p-value : 0.743235 > 0.05 이므로 귀무가설 채택, 대립가설 기각
# # 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하지 않는다.




          