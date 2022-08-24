# 문제2) 21세 이상의 피마 인디언 여성의 당뇨병 발병 여부에 대한 dataset을 이용하여 
# 당뇨 판정을 위한 분류 모델을 작성한다.

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
col_list = 'Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome'.split(',')
data.columns = col_list
print(data.head(2))
print(data.info())

x_data = data.drop('Outcome', axis=1)
y_data = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
print(x_train.shape, x_test.shape)      # (537, 8) (231, 8)

# Sequential API
model = Sequential()
model.add(Dense(36, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

import os
MODEL_DIR = './model2_sav/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

chkpoint = ModelCheckpoint(filepath=MODEL_DIR + "def.hdf5", monitor='val_loss', mode='auto',
                            verbose=0, save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='auto', patience=5)

history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=1000, verbose=2,
                    validation_split=0.2,
                    callbacks=[chkpoint, es])
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print('모델 분류 정확도: {:5.2f}%'.format(100*acc))
print('모델 분류 loss: {}'.format(loss))

# 새로운 값으로 예측
from keras.models import load_model
del model
mymodel = load_model(MODEL_DIR + "def.hdf5")
new_data = x_test[:3]
pred = mymodel.predict(new_data)
print('pred : ', np.where(pred > 0.5, 1, 0).flatten())


print('------------')
# Function API
inputs = Input(shape=((x_data.shape[1],)))
output1 = Dense(36, activation='relu')(inputs)
output2 = Dense(8, activation='relu')(output1)
output3 = Dense(1, activation='sigmoid')(output2)
model2 = Model(inputs, output3)

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model2.summary())

history2 = model2.fit(x=x_train, y=y_train, batch_size=x_train.shape[1], shuffle=True, epochs=50, verbose=2)
loss, acc = model2.evaluate(x_test, y_test, verbose=2)
print('모델 분류 정확도: {:5.2f}%'.format(100*acc))
print('모델 분류 loss: {}'.format(loss))

# 시각화
loss = history2.history['loss']
acc = history2.history['acc']

epoch_len = np.arange(len(loss))

plt.plot(epoch_len, loss, c='b', label='loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'best')
plt.show()

plt.plot(epoch_len, acc, c='r', label='acc', )
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(loc = 'best')
plt.show()


# 예측
col_list.remove('Outcome')
print(col_list)
new_data = []

# 키보드로 값 받기
# ex) 6 148 72 35 0 33.6 0.62 50   실제값:1
for v in col_list:
    new_data.append(float(input('{} :'.format(v))))
print('new_data : ', new_data)

# 차원확대
np_new_data = np.array(new_data).reshape(-1, len(new_data))
print('new_data : ', np_new_data)

pred = model2.predict(np_new_data)
print(np.where(pred > 0.5, '당뇨 환자', '당뇨 환자 아님').flatten())
