# 일원분산분석
# 강남구에 있는 GS편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 있는가 검정하기

# 귀무 : GS편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 없다.
# 대립 : GS편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import urllib.request

url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3.txt'
# data = pd.read_csv(url, header=None)
# print(data.head(2), type(data))  # <class 'pandas.core.frame.DataFrame'>
# print(data.describe())

data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')
print(data[:2], type(data))  # <class 'numpy.ndarray'> [[243.   1.]

# 세 개의 집단에 급여 자료의 평균
gr1 = data[data[:, 1] == 1, 0]
gr2 = data[data[:, 1] == 2, 0]
gr3 = data[data[:, 1] == 3, 0]
print(gr1, ' ', np.mean(gr1))  # 316.625
print(gr2, ' ', np.mean(gr2))  # 256.444
print(gr3, ' ', np.mean(gr3))  # 278.0

print('정규성 확인')
print(stats.shapiro(gr1).pvalue)  # 만족
print(stats.shapiro(gr2).pvalue)
print(stats.shapiro(gr3).pvalue)

print('등분산성 확인')
# print(stats.levene(gr1, gr2, gr3).pvalue)  # 표본의 갯수가 적으므로 적당하지 않음
print(stats.bartlett(gr1, gr2, gr3).pvalue)  # 0.35080 > 0.05 만족

# 데이터의 산포도 
plt.boxplot([gr1, gr2, gr3], showmeans=True)
plt.show()

# 일원분산분석 방법1 
df = pd.DataFrame(data, columns=['pay', 'group'])
print(df.head(3))
lmodel = ols('pay ~ C(group)', data=df).fit()
print(anova_lm(lmodel, typ=1))  # 그룹은 범주형이라는 사실을 알려주기 위해 C를 적어준다.
# 해석 : 0.043589 < 0.05 이므로 귀무가설 기각, 대립가설 채택
# GS편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 있다.

print()
# 일원분산분석 방법2
f_statistic, p_value = stats.f_oneway(gr1, gr2, gr3)
print('f_statistic, p_value:', f_statistic, p_value)
# f_statistic, p_value: 3.711335  0.043589

# 사후 검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukeyResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tukeyResult)

tukeyResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()  