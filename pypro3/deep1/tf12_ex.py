# Regression 
# https://github.com/pykwon/python/tree/master/data
# 자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 대여횟수에 영향을 주는 변수들을 골라 다중선형회귀분석 모델을 작성하시오.
# 모델 학습시에 발생하는 loss를 시각화 하고 설명력을 출력하시오.
# 새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대한 대여횟수 예측결과를 콘솔로 출력하시오.

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/data/train.csv', sep=',')
del data['datetime'], data['casual'], data['registered']
print(data.head(3))
print(data.info())

# count와의 상관관계
print(data.corr()['count'])  # temp, atemp, humidity

# feature, label
x = data[['temp', 'humidity']]
y = data[['count']]
print(x[:3])
print(y[:3])

print()
# train/ test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=123)
print(train_x.head(2))

print('--------------------------')
model = Sequential()
model.add(Dense(units=10, input_dim=3, activation='linear'))
model.add(Dense(units=5, activation='linear'))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(train_x, train_y, epochs=100, verbose=0)
print('train/test를 한 evaluate : ', model.evaluate(test_x, test_y))

print()
print(test_x[:5])  # 임의의 자료 하나를 골라 값 비교
print('실제값:', test_y[5])
print('예측값:', model.predict(test_x[:5]).flatten())

# 결정계수
from sklearn.metrics import r2_score
pred = model.predict(test_x)
print('train/test를 한 r2_score:', r2_score(test_y, pred))

# 시각화
import matplotlib.pyplot as plt 
plt.plot(test_y, 'b', label='real')
plt.plot(pred, 'r--', label='predict')
plt.show()


