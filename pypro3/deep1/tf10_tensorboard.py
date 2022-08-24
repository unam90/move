# TensorBoard는 TensorFlow에 기록된 로그를 그래프로 시각화시켜서 보여주는 도구다.

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorboard

# 5명이 세번의 시험 점수로 다음번 시험 점수 예측
x_data = np.array([[70, 85, 80],[71, 89, 78],[50, 80, 60],[66, 20, 60],[50, 30, 10]])
print(x_data)
y_data = np.array([73, 82, 72, 55, 34])

model = Sequential()
# model.add(Dense(1, input_dim=3, activation='linear'))
model.add(Dense(6, input_dim=3, activation='linear', name='a'))  # layer가 복잡해질 때는 name을 주는 게 좋다.
model.add(Dense(3, activation='linear', name='b'))
model.add(Dense(1, activation='linear', name='c'))

print(model.summary())

opti = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer = opti, loss='mse', metrics=['mse'])

# 텐서보드 객체 만들기
from keras.callbacks import TensorBoard
tb = TensorBoard(log_dir='.\\my', histogram_freq=1, 
                 write_graph=True, 
                 write_images=True,
                 update_freq='epoch',
                 profile_batch=2,
                 embeddings_freq=1)

history = model.fit(x_data, y_data, batch_size=1, epochs=50, verbose=0,
                    callbacks=[tb])

plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

loss_metrics = model.evaluate(x=x_data, y=y_data)
print('loss_metrics:', loss_metrics)

from sklearn.metrics import r2_score
print('설명력:', r2_score(y_data, model.predict(x_data)))

new_x_data = np.array([[44, 55, 10],[95, 55, 78]])
print('새로운 예측 결과:', model.predict(new_x_data).flatten())



