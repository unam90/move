# Keras 모듈을 사용해 DeepLearning 모델 네트워크 구성 샘플
# 논리회로 분류 

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])  # xor

model = Sequential()
model.add(Dense(units=5, input_dim=2, activation='relu'))  # Dense : 완전연결층 (레이어의 갯수) / 벡터곱 병렬 연산 
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())
# 파라미터 수 = (입력 자료수 + 1) * 출력수 = 
# (input_dim갯수 + 바이어스 수 1개 * units갯수) = (2+1)*5, (5+1)*5, (5+1)*1

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x, y, epochs=100, batch_size=1, verbose=2)  
# batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 (기본값은 32)
# 예를 들어 10문제에 batch_size를 2를 주면 2문제 풀고 답을 맞춘다. 5번의 가중치 갱신이 일어난다.
# epochs : 학습량 / 학습량이 높으면 정확도가 높아지지만 overfitting이 생길 수도 있다.
print(history.history)  # 진행되는 과정의 loss와 accuracy를 변수에 담아서 확인
print(history.history['loss'])  # 진행되는 과정의 loss값 확인
print(history.history['accuracy'])  # 진행되는 과정의 accuracy값 확인
# loss가 줄어들면 accuracy 증가

# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

print(model.weights)  # 가중치와 bias값 확인 가능

print()
pred = (model.predict(x) > 0.5).astype('int32')
print('예측 결과 :', pred.flatten())








