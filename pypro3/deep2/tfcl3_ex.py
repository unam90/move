# 문제2) 21세 이상의 피마 인디언 여성의 당뇨병 발병 여부에 대한 dataset을 이용하여 당뇨 판정을 위한 분류 모델을 작성한다.
# 피마 인디언 당뇨병 데이터는 아래와 같이 구성되어있다.
#   Pregnancies: 임신 횟수
#   Glucose: 포도당 부하 검사 수치
#   BloodPressure: 혈압(mm Hg)
#   SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
#   Insulin: 혈청 인슐린(mu U/ml)
#   BMI: 체질량지수(체중(kg)/키(m))^2
#   DiabetesPedigreeFunction: 당뇨 내력 가중치 값
#   Age: 나이
#   Outcome: 5년 이내 당뇨병 발생여부 - 클래스 결정 값(0 또는 1)
# 당뇨 판정 칼럼은 outcome 이다.   1 이면 당뇨 환자로 판정
# train / test 분류 실시
# 모델 작성은 Sequential API, Function API 두 가지를 사용한다.
# loss, accuracy에 대한 시각화도 실시한다.

import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import optimizers, Input


data = pd.read_csv('../testdata/pima-indians-diabetes.data.csv', header=None)
print(data.head(2))
print(data.info())
# print(len(data[data.iloc[:, 8] == 0]))  # 263
# print(len(data[data.iloc[:, 8] == 1]))  # 496 (당뇨환자 수)
#
# dataset = data.values
# print(dataset[:2], type(dataset))
#
# x = dataset[:, 0:7]  # 독립변수
# y = dataset[:, -1]   # 종속변수
# print(x[:2])
# print(y[:2])
#
# # train / test
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#
# # model 생성
# # 1) Sequential API
# model = Sequential()
# model.add(Dense(32, input_shape=(x_train.shape[1],), activation='relu', kernel_initializer='he_normal'))
# model.add(BatchNormalization())
# model.add(Dense(16, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(8, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#
# print(model.summary())
#
# es = EarlyStopping(monitor='val_loss', mode='auto', patience=5)
#
# history = model.fit(x_train, y_train, epochs=10000, batch_size=32, 
#                     verbose=2, callbacks=[es])
#
# loss, acc = model.evaluate(x_test, y_test, verbose=2)
# print('모델 정확도 : {:5.2f}%'.format(100 * acc))
# print('모델 loss : {:5.2f}%'.format(loss))
#
# # vloss = history.history['val_loss']
# # loss = history.history['loss']
# # vacc = history.history['accuracy']
# # acc = history.history['accuracy']
#
# # # 2) function API
# # from keras.layers import Input, Dense
# # from keras.models import Model
# #
# # inputs = Input(shape=(1,))
# # output1 = Dense(32, activation='sigmoid')(inputs)
# # outputs = Dense(1, activation='sigmoid')(output1)
# # model2 = Model(inputs, outputs)
# #
# # model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # model2.fit(x_train, y_train, batch_size=32, epochs=1000, shuffle=False, verbose=1)
# # loss2 = model2.evaluate(x_train, y_train, batch_size=32, verbose=0)
# # print('evaluate2:', loss2)