# feature 간 단위의 차이가 클 경우 정규화/표준화 작업이 효과적

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
import pandas as pd
import numpy as np

data = pd.read_csv('../testdata/Advertising.csv')
print(data.head(2))
del data['no']
print(data.head(2))

np.random.seed(123)

# 정규화 
# scaler = MinMaxScaler(feature_range=(0, 1))
# xy = scaler.fit_transform(data)
# print(xy[:2])

print()
xy = minmax_scale(data, axis=0, copy=True)  # 원본 데이터는 보존
print(data.head(2))
print(xy[:2], len(xy))

# train / test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xy[:, 0:-1], xy[:, -1], shuffle=True, 
                                                    test_size=0.3, random_state=123)  # 시계열데이터는 shuffling을 하면 안된다.
print(x_train[:2], x_train.shape)
print(y_train[:2], y_train.shape)

model = Sequential()
model.add(Dense(20, input_dim=3, activation='linear')) 
model.add(Dense(10, activation='linear')) 
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(model.summary()) 

import tensorflow as tf
tf.keras.utils.plot_model(model, 'aaa.png')

history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
          validation_split=0.2) # train data를 다시 7:3으로 쪼개서 검증 (overfitting 방지가 목적)  
        # validation_data=(x_test, y_test) 검증도 같이 하기

loss = model.evaluate(x_test, y_test)
print('loss:', loss)
print('loss:', loss[0])

print('history:', history.history)
print(history.history['loss'])
print(history.history['mse'])
print(history.history['val_loss'])
print(history.history['val_mse'])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

from sklearn.metrics import r2_score
print('r2_score:', r2_score(y_test, model.predict(x_test)))