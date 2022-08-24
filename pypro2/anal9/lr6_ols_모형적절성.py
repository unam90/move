# *** 선형회귀분석의 기존 가정 충족 조건 ***
# . 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
# . 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
# . 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
# . 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
# . 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.

# 여러 매체에 광고비 사용에 따른 판매량 데이터 
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tests.frame.methods.test_sort_values import ascending
plt.rc('font', family='malgun gothic')

advdf = pd.read_csv('Advertising.csv', usecols=[1,2,3,4]) # 1,2,3,4열만 가져오기
print(advdf.head(3), advdf.shape)
print(advdf.info())

print()
print('상관계수')
print(advdf.loc[:, ['sales', 'tv']].corr())  # sales와 tv의 상관관계 : 0.782224 강한 양의 상관관계

# 단순선형회귀
lm = smf.ols(formula='sales ~ tv', data=advdf).fit()
print(lm.summary())  
# Prob (F-statistic): 1.47e-42 < 0.05 이므로 모델에 적합
# R-squared:0.612 독립변수가 종속변수를 61프로 정도 설명

"""
# 시각화
plt.scatter(advdf.tv, advdf.sales)
plt.xlabel('tv')
plt.ylabel('sales')
y_pred = lm.predict(advdf.tv)
plt.plot(advdf.tv, y_pred, c='red')
plt.title('단순선형회귀')
plt.show()
"""

# 예측 : 새로운 tv값으로 sales를 예측 
x_new = pd.DataFrame({'tv':[222.2, 55.5, 100.0]})
pred = lm.predict(x_new)
print('예측값:', pred.values) # 예측값: [17.59523505  9.67087709 11.78625759]

# 다중선형회귀
print(advdf.corr())
# newspaper는 sales와의 상관관계가 약하다.

# lm_mul = smf.ols(formula='sales ~ tv + radio + newspaper', data=advdf).fit()
lm_mul = smf.ols(formula='sales ~ tv + radio', data=advdf).fit()
# newspaper를 제외하면 설명력이 높아진다.
print(lm_mul.summary())  
# 모델에서 확인한 결과 newspaper의 p-value값 0.860 > 0.05 이므로 
# newspaper는 독립변수에서 제외를 하는 것이 바람직하다.

# 예측 : 새로운 tv와 radio값으로 sales를 예측
x_new2 = pd.DataFrame({'tv':[222.2, 55.5, 100.0], 'radio':[30, 40, 50]})
pred2 = lm_mul.predict(x_new2)
print('예측값:', pred2.values) # 예측값: [18.72764663 12.98026122 16.89629275]

print('\n선형회귀분석의 기존 가정 충족 조건')

# 잔차 구하기
fitted = lm_mul.predict(advdf.iloc[:, 0:2])  # newspaper 제외하고 예측값 얻기
print(fitted)
residual = advdf['sales'] - fitted  # 실제값 - 예측값 = 잔차
print(residual[:10].values)
print(np.mean(residual))  # 잔차의 평균은 0에 가까움

import seaborn as sns
print('선형성')
sns.regplot(fitted, residual, lowess = True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='grey')
plt.show()  # 평평하지 않고 곡선 형태를 보이므로 선형성을 만족하지 못함. 다항회귀 추천

print('정규성 : Q-Q plot')
import scipy.stats 
sr = scipy.stats.zscore(residual)  # 표본에 있는 z값을 계산함 (확률분포)
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x, y)
plt.plot([-3, 3], [-3, 3], '--', color='blue')
plt.show()

# 잔차의 정규성을 샤피로 검정으로 확인
print('샤피로 검정 : ', scipy.stats.shapiro(residual))
# ShapiroResult(statistic=0.9180378317832947, pvalue=4.190036317908152e-09)
# pvalue=4.190036317908152e-09 < 0.05 이므로 정규성 만족하지 않음 
# log를 취하는 등의 작업으로 정규분포를 따르도록 데이터 가공 필요 

print('독립성: 잔차가 자기상관(인접 관측치의 오차가 상관되어 있음)이 있는지 확인')
# Durbin-Watson 검정을 함. 0 ~ 4 사이의 값이 나옴. 2에 가까우면 자기상관이 없음.
# Durbin-Watson:2.081 이므로 독립성은 만족

print('등분산성')
sns.regplot(fitted, np.sqrt(sr), lowess = True, line_kws={'color':'red'})
plt.show()  
# 빨간색 실선이 수평하지 못하므로 등분산성을 만족하지 못한다.
# 이상값, 비선형 확인 
# 정규성은 만족하나, 등분산성을 만족하지 않은 경우 가중회귀분석 추천  

print('다중공선성 : 독립변수 간 강한 상관관계 확인')
# 분산 팽창 인수(VIF, Variance Inflation Factor)로 확인
# 10을 넘으면 다중공선성이 발생하는 변수 
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(variance_inflation_factor(advdf.values, 1))  # tv        12.57031  
print(variance_inflation_factor(advdf.values, 2))  # radio     3.153498

print()
# 극단치 확인 : Cook's distance
from statsmodels.stats.outliers_influence import OLSInfluence

cd, _ = OLSInfluence(lm_mul).cooks_distance  # 극단값을 나타내는 지표를 반환
print(cd.sort_values(ascending=False).head())

# 극단치를 시각화해서 보기
import statsmodels.api as sm
sm.graphics.influence_plot(lm_mul, criterion='cooks')
plt.show()  # 원이 클수록 극단값

print(advdf.iloc[[130, 5, 35]])  # 제외하기를 권장하는 행 


 

