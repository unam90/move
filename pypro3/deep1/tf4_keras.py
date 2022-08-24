# Keras 모듈을 사용해 DeepLearning 모델 네트워크 구성 샘플
# 논리회로 분류 

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation

# 케라스 모델링 순서 
# 1. 데이터 셋 생성
#    원본 데이터를 불러오거나 데이터를 생성한다.
#    데이터로부터 훈련셋, 검증셋, 시험셋을 생성한다. 이 때 딥러닝 모델의 학습 및 평가를 할 수 있도록 포맷 변환을 한다.
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])  # or
# y = np.array([0,1,1,0])  # xor
print(x)  # feature
print(y)  # label

# 2. 모델 구성
#   시퀀스 모델을 생성한 뒤 필요한 레이어를 추가하며 구성한다. 좀 더 복잡한 모델이 필요할 때는 케라스 함수 API를 이용한다.
# model = Sequential([
#     Dense(input_dim=2, units=1), # Dense : node들을 담아놓는 layer
#     Activation('sigmoid')
# ])

model = Sequential()
# model.add(Dense(units=1, input_dim=2))
# model.add(Activation('sigmoid'))
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))


# 3. 모델 학습 과정 설정(compile)
#   학습하기 전, 학습에 대한 설정을 수행한다. 손실 함수 및 최적화 방법을 정의. compile() 함수를 사용한다.
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# 4. 모델 학습시키기
#   훈련셋을 이용하여 구성한 모델로 학습 시킨다. fit() 함수를 사용한다.
model.fit(x, y, epochs=500, batch_size=1, verbose=0) # epochs: 학습횟수, verbose: 학습진행과정 보여주는 정도

# 5. 모델 평가
#   준비된 시험셋으로 학습한 모델을 평가한다. eval‎uate() 함수를 사용
loss_metrics = model.evaluate(x, y)
print(loss_metrics)

print()
# 6. 모델 사용하기 : 예측값 얻기
#   임의의 입력으로 모델의 출력을 얻는다. predict() 함수를 사용한다.
# pred = model.predict(x)
pred = (model.predict(x) > 0.5).astype('int32')
print('예측값:', pred.flatten())
print('실제값:', y)

# 7. 모델 저장
model.save('tf1.hdf5')
del model

from keras.models import load_model
model2 = load_model('tf1.hdf5')
print('예측값:', (model2.predict(x) > 0.5).astype('int32').flatten())










