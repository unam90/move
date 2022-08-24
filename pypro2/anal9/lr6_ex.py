# 회귀분석 문제 3) 
# kaggle.com에서 carseats.csv 파일을 다운 받아 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.

import statsmodels.api
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tests.frame.methods.test_sort_values import ascending
plt.rc('font', family='malgun gothic')

carseat = pd.read_csv('Carseats.csv', usecols=[0,3,5])
print(carseat.head(3), carseat.shape)
print(carseat.info())

print()
# 상관관계 확인
print(carseat.corr())
# 상관계수
print(carseat.loc[:, ['Sales', 'Price']].corr())  # -0.444951 음의 상관관계

# 단순선형회귀
lm = smf.ols(formula='Sales ~ Price', data=carseat).fit()
print(lm.summary())
# Prob (F-statistic): 4.38e-08 < 0.05 이므로 모델에 적합

"""
# 시각화
plt.scatter(carseat.Price, carseat.Sales)
plt.xlabel('price')
plt.ylabel('sales')
y_pred = lm.predict(carseat.Price)
plt.plot(carseat.Price, y_pred, c='red')
plt.show()
"""

# 다중선형회귀
mul_lm = smf.ols(formula = 'Sales ~ Price + Advertising', data=carseat).fit()
print(mul_lm.summary())

print()
# 회귀분석모형의 적절성을 위한 조건 체크
# 잔차 구하기
fitted = lm.predict(carseat.iloc[:, 1:3])
print(fitted)
residual = carseat['Sales'] - fitted
print(residual[:10].values)
print(np.mean(residual)) # 잔차의 평균은 0에 가까움

import seaborn as sns
# 선형성
sns.regplot(fitted, residual, lowess = True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='grey')
plt.show()

# 정규성: Q-Q plot
import scipy.stats
sr = scipy.stats.zscore(residual)
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x, y)
plt.show()

# 잔차의 정규성을 샤피로 검정으로 확인
print(scipy.stats.shapiro(residual))
# ShapiroResult(statistic=0.9953006505966187, pvalue=0.27001717686653137)
# pvalue=0.270017 > 0.05 이므로 정규성 만족

# 독립성: 잔차가 자기상관(인접 관측치의 오차가 상관되어 있음)이 있는지 확인
# Durbin-Watson:1.892 : 2에 가까워 자기상관이 없음. 독립성 만족

# 등분산성
sns.regplot(fitted, np.sqrt(sr), lowess=True, line_kws={'color':'red'})
plt.show()
# 빨간색 실선이 수평에 가까우므로 등분산성을 만족한다.

# 다중공선성 : 독립변수 간 강한 상관관계 확인 
# 분산 팽창 인수(VIF, Variance Inflation Factor)로 확인
# 10을 넘으면 다중공선성이 발생하는 변수 
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(variance_inflation_factor(carseat.values, 1))  # Price        2.19353
print(variance_inflation_factor(carseat.values, 2))  # Advertising  4.77296
