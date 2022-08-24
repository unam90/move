# 파이썬이 지원하는 회귀분석 대표적인 방법 맛보기

# 회귀분석 : 독립변수(연속형)가 종속변수(연속형)에 얼마나 영향을 주는지 인과관계를 분석
# 정량적인 모델을 생성

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np
from daal4py.sklearn.linear_model._linear import LinearRegression
from astropy.units import yyr

print('방법1 : make_regression 사용. model X')
np.random.seed(12)
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print(x)
print(y)
print(coef)  # 기울기 : 89.47430739278907
# 수식(모델) : y = 89.47430739278907 * x + 100

# 모델에 의한 예측값
new_x = 0.567  # 한번도 경험하지 못한 새로운 x에 대한 y값을 얻음 
y_pred = 89.47430739278907 * new_x + 100
print('예측값은 ', y_pred)

xx = x 
yy = y

print()
print('방법2 : LinearRegression 사용. model O')
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# 모델 완성
fit_model = model.fit(xx, yy)  # 독립변수(x)와 종속변수(y)로 학습을 진행. 절편과 기울기를 반환
# 잔차가 최소화 되는 절편과 기울기를 반환한다.

print('기울기:', fit_model.coef_)
print('절편:', fit_model.intercept_)

new_x = 1.234
y_pred = 89.47430739 * new_x + 100
print('2. 예측값은 ', y_pred)
# 예측값 지원 함수를 사용 
y_pred = fit_model.predict([[new_x]])
print('2. 예측값은 ', y_pred)

print('방법3 : ols 사용. model O')  # ols : ordinary list square
import statsmodels.formula.api as smf
import pandas as pd
print(type(xx), xx.shape)  # <class 'numpy.ndarray'> (50, 1)
x1 = xx.flatten()  # 차원 축소
print(x1.shape)  # (50,) 1차원으로 바뀜
y1 = yy
print(y1.shape)
data = np.array([x1, y1])
df = pd.DataFrame(data.T)
df.columns = ['x1', 'y1']
print(df.head(3))  

model2 = smf.ols(formula='y1 ~ x1', data=df).fit()  # R의 형식을 흉내냄
# model2.fit()
print(model2.summary())  # 모델 정보 확인 
# Intercept(절편):100.0000, x1의 기울기(slope):89.4743 

new_df = pd.DataFrame({'x1':[-1.3, -0.5, 1.234]})
y_pred2 = model2.predict(new_df)
print('3. 예측값은', y_pred2)

print()
print('방법4 : linregress 사용. model O')
from scipy import stats

model3 = stats.linregress(x1, y1)
print('기울기:', model3.slope)  # 89.4743073927891
print('절편:', model3.intercept)  # 100.0





