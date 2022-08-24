# 비선형 회귀모델 : 선형 가정이 어긋날 때 (정규성을 만족하지 못함) 대처할 수 있는 방법으로 다항회귀모델 가능

# 입력 데이터의 특징을 변환해서 선형모델 개선
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])
# plt.scatter(x, y)
# plt.show()

# 선형회귀 모델 작성 
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis]  # 차원 확대 
print(x, x.shape)

print()
# 비선형 회귀 모델 작성
from sklearn.preprocessing import PolynomialFeatures  # 다항식 특징을 추가 가능
poly = PolynomialFeatures(degree=2, include_bias = False)  # degree=열갯수
x2 = poly.fit_transform(x)  # 특징 행렬 만듦
print(x2)

model2 = LinearRegression().fit(x2, y)
ypred2 = model2.predict(x2)
print(ypred2)

plt.scatter(x, y)
plt.plot(x, ypred2, c='red') 
plt.show()