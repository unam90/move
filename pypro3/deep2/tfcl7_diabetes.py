# 이항분류(sigmoid)는 다항분류(softmax)로 처리 가능

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataset = np.loadtxt('../testdata/diabetes.csv', delimiter=',')

print(dataset.shape)  # (759, 9)
print(dataset[:1])
print(set(dataset[:, -1]))  # 모든행에 마지막열 {0.0, 1.0}

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset[:, 0:8], dataset[:, -1],
                                                    test_size=0.3, random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (531, 8) (228, 8) (531,) (228,)

# 이항 분류
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
scores = model.evaluate(x_test, y_test)
print('%s:%.2f%%'%(model.metrics_names[1], scores[1] * 100))  # scores를 퍼센트로 표기 acc:77.63%
print('%s:%.2f'%(model.metrics_names[0], scores[0]))  # loss 보기 loss:0.50
pred = model.predict([[-0.34, 0.487437, 0.180328, -0.292929, 0., 0.00149028, -0.53117, -0.03]])
print('예측결과:', pred, ' ', np.where(pred > 0.5, 1, 0))  # pred값이 0.5보다 크면 1 아니면 0

print('---' * 20)
# 다항분류 -----------------------------
from keras.utils import to_categorical
# label을 원핫 처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[0])  # [1. 0.]

model2 = Sequential()
model2.add(Dense(64, input_dim=8, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(2, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])
model2.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
scores = model2.evaluate(x_test, y_test)  # batch_size 안쓰면 기본 32
print('%s:%.2f%%'%(model2.metrics_names[1], scores[1] * 100))  # scores를 퍼센트로 표기 acc:77.63%
print('%s:%.2f'%(model2.metrics_names[0], scores[0]))  # loss 보기 loss:0.50

pred = model2.predict([[-0.34, 0.487437, 0.180328, -0.292929, 0., 0.00149028, -0.53117, -0.03]])
print('예측결과:', pred, ' ', np.argmax(pred))