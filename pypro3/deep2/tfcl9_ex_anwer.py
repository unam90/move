# 문제3) BMI 식으로 작성한 bmi.csv 파일을 이용하여 분류모델 작성 후 분류 작업을 진행한다.
# train/test 분리 작업을 수행.
# 평가 및 정확도 확인이 끝나면 모델을 저장하여, 저장된 모델로 새로운 데이터에 대한 분류작업을 실시한다.
# EarlyStopping, ModelCheckpoint 사용.
# 새로운 데이터, 즉 키와 몸무게는 키보드를 통해 입력하기로 한다. fat, normal, thin으로 분류

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import numpy as np

data = pd.read_csv('../testdata/bmi.csv')
print(data.head(2))

x_data = data[['height','weight']]
y_data = data['label'].astype('category').values

#print(y_data.codes[:5])
print(y_data.categories)  # Index(['fat', 'normal', 'thin'], dtype='object')
y_data_codes = y_data.codes

print(x_data[:5])
print(y_data_codes[:5])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_codes, test_size=0.3, random_state=1)

print(x_train.shape, x_test.shape)  # (14000, 2) (6000, 2)
print(y_train.shape, y_test.shape)  # (14000,) (6000,)

# 모델 
model = Sequential()
model.add(Dense(48, activation='relu', input_dim=2))
model.add(Dense(12, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='loss', mode='auto', patience=10)

import os
model_dir = './model3_sav/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

modelpath = '{}bmi_model.hdf5'.format(model_dir)
chkpoint = ModelCheckpoint(filepath=modelpath, loss='loss', verbose=2, save_best_only=True)
"""
history = model.fit(x_train, y_train, epochs=5000, batch_size = 512,
                    validation_split=0.2, verbose=2, 
                    callbacks=[es, chkpoint])

print('eval : ', model.evaluate(x_test, y_test))

del model
"""
# predict
from keras.models import load_model
mymodel = load_model(model_dir + 'bmi_model.hdf5')

height = float(input('키 :'))
weight = float(input('몸무게 :'))
new_data = np.array([[height, weight]])
print(new_data)

pred = mymodel.predict(new_data)
print('결과 : ', y_data.categories[np.argmax(pred)])
