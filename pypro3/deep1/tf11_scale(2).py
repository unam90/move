# feature 간 단위의 차이가 클 경우 정규화/표준화 작업이 효과적
# 정규화/표준화는 label에는 하지 않는다.

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
import pandas as pd
import numpy as np

# StandardScaler : 기본 스케일러. 평균과 표준펀차를 사용. 이상치가 있으면 불균형
# MinMaxScaler  : 최대, 최소값이 각각 0, 1이 되도록 정규화. 이상치에 민감
# RobustScaler : 이상치의 영향을 최소화함. 중앙값과 사분위를 사용

data = pd.read_csv("../testdata/Advertising.csv")
print(data.head(2))
del data['no']
print(data.head(2))

fdata = data[['tv', 'radio', 'newspaper']]
# ldata = data[['sales']]
ldata = data.iloc[:,[3]]
print(fdata.head(2))
print(ldata.head(2))

np.random.seed(123)

# 정규화 : 기본적으로 0 ~ 1 사이의 값으로 변경된다.
# scaler = MinMaxScaler(feature_range=(0, 1))  # feature_range=(0, 1) 기본값
# fedata = scaler.fit_transform(fdata)
# print(fedata)

fedata = minmax_scale(fdata, axis=0, copy=True)  # 행기준, 원본 데이터는 보존
print(fdata.head(2))
print(fedata[:2], len(fedata))   # 200

# train / test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(fedata, ldata, shuffle=True,
                                                    test_size=0.3, random_state=123)
print(x_train[:2], x_train.shape)  # (140, 3)
print(y_train[:2], y_train.shape)  # (140, 1)

model = Sequential()
model.add(Dense(20, input_dim=3, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics = ['mse'])
print(model.summary())

import tensorflow as tf
tf.keras.utils.plot_model(model, 'aaa.png')

history = model.fit(x_train, y_train, epochs=100, 
                    batch_size = 32, verbose=2, 
                    validation_split=0.2)   # validation_data=(x_test, y_test)

# 모델 평가 score 보기
loss = model.evaluate(x_test, y_test)   # test data 사용
print('loss : ', loss)
print('loss : ', loss[0])

# history 값 확인
print('history : ', history.history)
print(history.history['loss'])
print(history.history['mse'])
print(history.history['val_loss'])
print(history.history['val_mse'])

# loss 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

from sklearn.metrics import r2_score
print('r2_score : ', r2_score(y_test, model.predict(x_test)))

# predict
pred = model.predict(x_test[:3])
print('예측값 : ', pred.flatten())
print('실제값 : ', y_test[:3].values.flatten())
