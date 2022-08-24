# 문제1)
# http://www.randomservices.org/random/data/Galton.txt
# data를 이용해 아버지 키로 아들의 키를 예측하는 회귀분석 모델을 작성하시오.
# train / test 분리
# Sequential api와 function api 를 사용해 모델을 만들어 보시오.
# train과 test의 mse를 시각화 하시오
# 새로운 아버지 키에 대한 자료로 아들의 키를 예측하시오.

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 전처리
data = pd.read_csv('http://www.randomservices.org/random/data/Galton.txt', sep='\t', usecols=['Father', 'Gender', 'Height'])
print(data.head(3))

# Gender열에서 M(아들)만 분리
son_data = data[data['Gender']=='M'].drop('Gender', axis=1)
print(son_data.head(2))

# 아버지 아들 키 분리 
father_x = son_data.Father
son_y = son_data['Height']
print('상관계수:', np.corrcoef(father_x, son_y))

# train / test
train_x, test_x, train_y, test_y = train_test_split(father_x, son_y, test_size=0.3, random_state=1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

print('1) Sequential API 사용')
model = Sequential()
model.add(Dense(units=5, input_dim=1, activation='linear'))
model.add(Dense(units=1, activation='linear'))
print(model.summary())

opti = optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])  

# 모델 학습
history = model.fit(x=train_x, y=train_y, batch_size=1, epochs=50, verbose=0)
loss_metrics = model.evaluate(x=test_x, y=test_y, verbose=0)
print('loss_metrics:', loss_metrics)

# from sklearn.metrics import r2_score
# print('설명력:', r2_score(test_y, model.predict(test_x)))  # 설명력이 안좋아서 제외함

print('실제값:', test_y.head().values)
print('예측값:', model.predict(test_x).flatten()[:5])

new_height=[90, 60, 80]
print('새로운 예측값:', model.predict(new_height).flatten())

# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.plot(train_x, model.predict(train_x), 'b', train_x, train_y, 'ko')  # train
plt.xlabel('아버지 키')
plt.ylabel('아들 키')
plt.show()

plt.plot(test_x, model.predict(test_x), 'b', test_x, test_y, 'ko')  # test
plt.xlabel('아버지 키')
plt.ylabel('아들 키')
plt.show()

# 학습 도중에 발생된 변화량을 시각화
plt.plot(history.history['mse'], label='평균제곱오차')
plt.xlabel('학습횟수')
plt.ylabel('mse')
plt.show()

print('2) function API 사용')
from keras.layers import Input
from keras.models import Model

# 각 층을 일종의 함수로써 처리를 함. 설계부분이 방법1과 다름
inputs = Input(shape=(1,))  # 입력 크기를 지정 (tuple로 작성)
output1 = Dense(units=5, activation='linear')(inputs)
outputs = Dense(2, activation='linear')(output1)

model2 = Model(inputs, outputs)
print(model2.summary())

opti = optimizers.Adam(learning_rate = 0.001)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])  
# mse(min-squared error) : 평균제곱오차(작을수록 좋다) / 추측값에 대한 정확성을 측정하는 방법

history2 = model2.fit(x=train_x, y=train_y, batch_size=4, epochs=50, verbose=0)

loss_metrics2 = model2.evaluate(x=test_x, y=test_y, verbose=0)
print('loss_metrics:', loss_metrics2)

# from sklearn.metrics import r2_score
# print('설명력:', r2_score(test_y, model2.predict(test_x)))  # 설명력이 낮아서 제외
print('실제값:', test_y.head().values)
print('예측값:', model2.predict(test_x).flatten()[:5])
print()
new_height2 = [90, 60, 80]
print('새로운 예측값:', model2.predict(new_height2).flatten())



