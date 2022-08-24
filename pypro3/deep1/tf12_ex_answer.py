# Regression 
# https://github.com/pykwon/python/tree/master/data
# 자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 대여횟수에 영향을 주는 변수들을 골라 다중선형회귀분석 모델을 작성하시오.
# 모델 학습시에 발생하는 loss를 시각화 하고 설명력을 출력하시오.
# 새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대한 대여횟수 예측결과를 콘솔로 출력하시오.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/data/train.csv', parse_dates=['datetime'])
print(data.head(2))
print(data.info())

# count와의 상관관계
print(data.corr()['count'])
# feature
feature = data[['temp','humidity','casual','registered']]
print(feature.head(2))

label = data[['count']]
print(label.head(2))

print(feature.shape, ' ', label.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature, label, shuffle=False,
                                                    test_size=0.3, random_state=1)

print(x_train.head(2))
# 모델
model = Sequential()
model.add(Dense(units=1, input_dim=feature.shape[1], activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(model.summary())

print()
history = model.fit(x_train, y_train, epochs=20, 
                    batch_size=32, validation_split = 0.3,
                    verbose=2)

print('evaluate : ', model.evaluate(x_test, y_test))

plt.plot(history.history['loss'], label = 'loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

print('예측값 :', model.predict(x_test[:5]).flatten())
print('실제값 :', y_test.head().values.flatten())
from sklearn.metrics import r2_score
print('설명력 :', r2_score(y_test, model.predict(x_test)))

print("'temp','humidity','casual','registered'")
temp = float(input('온도 :'))
humidity = int(input('습도 :'))
casual = int(input('비회원 대여량 :'))
registered = int(input('회원 대여량 :')) 

new_data = pd.DataFrame(
    {'temp':[temp], 'humidity':[humidity], 'casual':[casual], 'registered':[registered]})
print(new_data)

pred = model.predict(new_data)
print('예상 count는 {}'.format(pred[0].flatten()))
