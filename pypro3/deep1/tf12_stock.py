# 시계열 데이터로 다중회귀분석
# 주식데이터를 사용하여 하루 전 데이터로 다음날 종가를 예측

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 배열자료 형태로 읽기
xy = np.loadtxt('../testdata/stockdaily.csv', delimiter=',', skiprows=1) 
# delimiterf: 구분자 / skiprows: 배열자료 형태로 출력이므로 문자열인 첫째행은 제외해줘야함
print(xy[:2], len(xy))

# 정규화
scaler = MinMaxScaler(feature_range=(0,1))
xy = scaler.fit_transform(xy)  # scaler.inverse_transform(xy) 역정규화
print(xy[:2])

print()
x_data = xy[:, 0:-1]  # 모든 행, 0열부터 마지막 열 전까지만
y_data = xy[:, [-1]]
print(x_data[:2])
print(y_data[:2])

print()
# 전날 데이터로 다음날 종가를 예측해야 하므로 데이터 가공 필요
print(x_data[0], y_data[0])
print(x_data[1], y_data[1])
x_data = np.delete(x_data, -1, axis=0) # 마지막 행 삭제  
y_data = np.delete(y_data, 0) # 0행 삭제
print()
print(x_data[0], y_data[0])

print('--------------------------')
model = Sequential()
model.add(Dense(units=1, input_dim=4, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(x_data, y_data, epochs=200, verbose=0)
print('train/test 안한 evaluate : ', model.evaluate(x_data, y_data))

print()
print(x_data[10])  # 임의의 자료 하나를 골라 값 비교 
test = x_data[10].reshape(-1, 4)
print('실제값:', y_data[10])
print('예측값:', model.predict(test).flatten())

# 결정계수
from sklearn.metrics import r2_score
pred = model.predict(x_data)
print('train/test 안한 r2_score:', r2_score(y_data, pred))  # 과적합 의심

# 시각화
plt.plot(y_data, 'b', label='real')
plt.plot(pred, 'r--', label='predict')
plt.show()

print('\n------과적합 방지를 목적으로 train / test split------------')
print(len(x_data))
train_size = int(len(x_data) * 0.7)  # 시계열 자료는 섞지 않는다.
test_size = len(x_data) - train_size
print(train_size, ' ', test_size)

print()
x_train, x_test = x_data[0:train_size], x_data[train_size:len(x_data)]
y_train, y_test = y_data[0:train_size], y_data[train_size:len(x_data)]
print(x_train[:2], x_train.shape, ' ', x_test[:2], x_test.shape)  # (511, 4) (220, 4)
print(y_train[:2], y_train.shape, ' ', y_test[:2], y_test.shape)  # (511,) (220,)


print('--------------------------')
model2 = Sequential()
model2.add(Dense(units=1, input_dim=4, activation='linear'))

model2.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model2.fit(x_train, y_train, epochs=200, verbose=0)
print('train/test를 한 evaluate : ', model2.evaluate(x_test, y_test))

print()
print(x_test[10])  # 임의의 자료 하나를 골라 값 비교 
print('실제값:', y_test[10])
print('예측값:', model2.predict(x_test[10]).flatten())

# 결정계수
pred2 = model2.predict(x_test)
print('train/test를 한 r2_score:', r2_score(y_test, pred2))

# 시각화
plt.plot(y_test, 'b', label='real')
plt.plot(pred2, 'r--', label='predict')
plt.show()

print('--------------------')
model3 = Sequential()
model3.add(Dense(units=1, input_dim=4, activation='linear'))

model3.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model3.fit(x_train, y_train, epochs=200, verbose=0, validation_split=0.15)
print('train/test => validation 후 evaluate : ', model3.evaluate(x_test, y_test))

# 결정계수
pred3 = model3.predict(x_test)
print('train/test => validation 후 r2_score:', r2_score(y_test, pred3))


# 머신러닝의 이슈 : 모델의 성능에 있어, 최적화와 일반화 사이의 줄다리기 



