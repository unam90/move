# 현대자동차 가격 예측을 위한 다중선형회귀 분석

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import tensorflow as tf

train_df = pd.read_excel('https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx?raw=true', sheet_name='train')
test_df = pd.read_excel('https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx?raw=true', sheet_name='test')
print(train_df.head(1))
print(test_df.head(1))

x_train = train_df.drop(['가격'], axis=1)  # 가격을 제외한 나머지 열은 feature
x_test = test_df.drop(['가격'], axis=1)  
y_train = train_df[['가격']]  # 가격 열을 label로 사용
y_test = test_df[['가격']] 

print(x_train.head(2))
print(y_train.head(2))
 
print(x_train.columns)
print(x_train.shape)  # (71, 10)
# print(x_train.describe())
print(set(x_train.종류))  # {'대형', '중형', '소형', '준중형'}
print(set(x_train.연료))  # {'LPG', '디젤', '가솔린'}
print(set(x_train.변속기)) # {'수동', '자동'}

# 범주형 칼럼들을 dummy변수로 만들기 : LabelEncoder, OneHotEncoder
# make_column_transformer를 사용하여 특정 열에만 OneHotEncoder 적용
transformer = make_column_transformer((OneHotEncoder(), ['종류', '연료', '변속기']),
                                      remainder='passthrough')
# remainder = 'passthrough' or 'drop' : passthrough 지정된 이외의 모든 열이 transformer로 전달
transformer.fit(x_train)
# 종류, 연료, 변속기 세 개의 칼럼이 참여해 OneHot처리됨
# [[0. 0. 1. 0. 0. 1. 0. 0. 1.]... 

x_train = transformer.transform(x_train)  # 모든 칼럼이 표준화됨
x_test = transformer.transform(x_test)
print('make_column_transformer:', type(x_train), x_train[:2])
print(x_train.shape)  # (71, 16)
print(y_train.shape)  # (71, 1)

# model 만들기
input = tf.keras.layers.Input(shape=(16,))
net = tf.keras.layers.Dense(units=32, activation='relu')(input)
net = tf.keras.layers.Dense(units=32, activation='relu')(net)
net = tf.keras.layers.Dense(units=1)(net)
model = tf.keras.models.Model(input, net)

print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=2)
print('evaluate:', model.evaluate(x_test, y_test))

y_pred = model.predict(x_test)
print('예측값:', y_pred[:10].flatten())
print('실제값:', y_test[:5].values.flatten())

# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['mse'], 'r', label='mse')
plt.plot(history.history['val_mse'], 'b', label='val_mse')
plt.xlabel('epochs')
plt.legend()
plt.show()
