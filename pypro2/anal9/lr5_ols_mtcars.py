# 선형회귀 : mtcars dataset 사용. ols() 사용
# ML 중 지도학습 : 귀납법적 추론방식을 사용 - 일반 사례를 수집해 법칙을 만듬

import statsmodels.api
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars)
print(mtcars.columns)
# print(mtcars.describe())
# print(mtcars.info())

print('상관관계 확인')
print(mtcars.corr())
print(np.corrcoef(mtcars.hp, mtcars.mpg))  # -0.776168
print(np.corrcoef(mtcars.wt, mtcars.mpg))  # -0.867659

print('단순선형회귀-----')
model1 = smf.ols('mpg ~ hp', data=mtcars).fit()
print(model1.summary())
# R-squared 0.602: 독립변수가 종속변수를 60% 정도 설명할 수 있다.
print(model1.summary().tables[0])
# y =  -0.0682 * x + 30.0989

print('다중선형회귀-----')
model2 = smf.ols('mpg ~ hp + wt', data=mtcars).fit()
print(model2.summary())
# y = -0.0318 * x1(hp) + -3.8778 * x2(wt) + 37.2273

# 예측 : 
new_pred = model2.predict({'hp':[110, 150, 90], 'wt':[4, 8, 2]})
print(new_pred)








