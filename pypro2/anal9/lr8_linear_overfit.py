# 선형회귀모델의 과적합 방지용 클래스 
# Linear Regression의 기본 알고리즘에 오버피팅 방지 목적의 제약조건을 담은 Ridge, Lasso, ElasticNet 회귀모형이 있다.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

iris = load_iris()
print(iris)
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target
iris_df["target_names"] = iris.target_names[iris.target]
print(iris_df[:3])

# 모델학습은 train dataset으로 모델 검정은 test dataset으로 하기
# train dataset, test dataset으로 나누기
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(iris_df, test_size = 0.3,random_state=12)  # 7 : 3 # random_state 랜덤한 값을 고정
print(len(train_set), len(test_set))  # 105 45

print('-------------------------------------')
# 회귀분석 방법 1 - LinearRegression
from sklearn.linear_model import LinearRegression
print(train_set.iloc[:, [2]])  # petal length (cm), 독립변수
print(train_set.iloc[:, [3]])  # petal width (cm), 종속변수

# 학습은 train dataset 으로 작업
model_linear = LinearRegression().fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])
print('slope : ', model_linear.coef_)  # 기울기 : 0.42259168
print('bias : ', model_linear.intercept_)  # 절편 : -0.39917733

# 모델 평가는 test dataset 으로 작업
pred = model_linear.predict(test_set.iloc[:, [2]])
print('예측값 : ', np.round(pred[:5].flatten(),1))
print('실제값 : ', test_set.iloc[:, [3]][:5].values.flatten())

from sklearn.metrics import r2_score
print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred)))  # 0.93833

print('\nRidge -----------')
# 회귀분석 방법 - Ridge: alpha값을 조정(가중치 제곱합을 최소화)하여 과대/과소적합을 피한다. 다중공선성 문제 처리에 효과적.
from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

#점수
print(model_ridge.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]]))  # 0.91880
print(model_ridge.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))    # 0.94101
pred_ridge = model_ridge.predict(test_set.iloc[:, [2]])
print('ridge predict : ', pred_ridge[:5])

# plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='red')
# plt.plot(test_set.iloc[:, [2]], model_ridge.predict(test_set.iloc[:, [2]]))
# plt.show()

print('\nLasso -----------')
# 회귀분석 방법 - Lasso: alpha값을 조정(가중치 절대값의 합을 최소화)하여 과대/과소적합을 피한다.
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.1).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

#점수
print(model_lasso.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])) # 0.913863
print(model_lasso.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))   # 0.940663
pred_lasso = model_lasso.predict(test_set.iloc[:, [2]])
print('lasso predict : ', pred_lasso[:5])

# plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='blue')
# plt.plot(test_set.iloc[:, [2]], model_lasso.predict(test_set.iloc[:, [2]]))
# plt.show()

# 회귀분석 방법 4 - Elastic Net 회귀모형 : Ridge + Lasso의 형태로 가중치 절대값의 합(L1)과 제곱합(L2)을 동시에 제약 조건으로 가지는 모형

print('\nElastic Net -----------')
# 회귀분석 방법 - ElasticNet: alpha값을 조정(가중치 절대값의 합을 최소화)하여 과대/과소적합을 피한다.
from sklearn.linear_model import ElasticNet
model_elastic = ElasticNet(alpha=0.1).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

#점수
print(model_elastic.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])) # 0.913863
print(model_elastic.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))   # 0.940663
pred_elastic = model_elastic.predict(test_set.iloc[:, [2]])
print('ElasticNet predict : ', pred_elastic[:5])

# plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]], color='cyan')
# plt.plot(test_set.iloc[:, [2]], model_elastic.predict(test_set.iloc[:, [2]]))
# plt.show()

