# 선형회귀모델 작성 : LinearRegression 클래스 사용

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print(mtcars.corr(method='pearson'))

print()
# hp가 mpg에 영향을 준다고 가정하고 선형회귀모델을 작성
# sklearn 제공 모델의 경우 독립변수(x, feature)는 2차원, 종속변수(y, label)는 1차원 배열 형식을 갖는다. 

x = mtcars[['hp']].values
print(x[:3], x.shape)  # hp열 자료를 2차원 배열 형식으로 추출

y = mtcars['mpg'].values
print(y[:3], y.shape)  # mpg열 자료를 1차원 배열 형식으로 추출

# plt.scatter(x, y)
# plt.show()

lmodel = LinearRegression().fit(x, y)  # 학습 후 모델 생성
# lmodel.fit(x, y)
# print(lmodel.summary())  # 'LinearRegression' object has no attribute 'summary'
print('회귀계수(slope): ', lmodel.coef_)  # 기울기
print('회귀계수(slope): ', lmodel.intercept_) # y절편
# y = -0.06822828 * x + 30.09886054

pred = lmodel.predict(x)
print('예측값 :', np.round(pred[:10], 1))
print('실제값 :', y[:10])

print('선형회귀모델 성능을 파악하기 위한 방법')
print('RMSE(평균제곱근오차):', mean_squared_error(y, pred))  # 13.9898
print('결정계수(설명력, r2_score) :', r2_score(y, pred))      # 0.6024

print()
# 한번도 경험하지 않은 hp 새로운 값으로 mpg를 예측
new_hp = [[120]]
new_pred = lmodel.predict(new_hp)
print('%s 마력인 경우 연비는 약 %s입니다'%(new_hp[0][0], new_pred[0])) 


