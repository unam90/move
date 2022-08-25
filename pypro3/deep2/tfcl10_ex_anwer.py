# 문제4) python/testdata_utf8/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.

import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../testdata/HR_comma_sep.csv')
print(data.head(2))
print(data.info())
# 이항 데이터 제거
data.drop(['left','promotion_last_5years','Work_accident'],axis = 1, inplace = True)

# x 전처리 sales category화, salary drop
pre1_x_data = data.drop('salary', axis=1)  # feature에서 제거
# pandas는 문자열을 object라는 자료형 사용. 문자열 값의 종류가 제한적일 때는 category를 사용할 수 있다. 메모리 절감 효과
sales_cate_data = pre1_x_data['sales'].astype('category')
print(sales_cate_data[:2])

x_data = pre1_x_data
x_data['sales'] = sales_cate_data.values.codes   # values.codes : category를 dummy화
print(x_data[:2])

# salary열은 label  - category 화 : LabelEncoder도 사용 가능
pre_y_data = data['salary'].astype('category').values
print(pre_y_data[:2])
print(pre_y_data.categories)    # Index(['high', 'low', 'medium'], dtype='object')
print(set(pre_y_data))  # {'high', 'low', 'medium'}
y_data = pre_y_data.codes
print(y_data[:5])    # [1 2 2 1 1 1 1 1 1 1]
print(set(y_data))   # {0, 1, 2}

# train/test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (10499, 6) (4500, 6) (10499,) (4500,)

# one-hot 
onehot_train_labels = tf.keras.utils.to_categorical(y_train)    # label에 대해 원핫 인코딩
onehot_test_labels = tf.keras.utils.to_categorical(y_test)
print(onehot_train_labels[:2])
print(onehot_test_labels[:2])

# Randomforest -----------------------
from sklearn.ensemble import RandomForestClassifier

rnd_model = RandomForestClassifier(n_estimators=500, criterion='entropy')
rnd_model.fit(x_train, onehot_train_labels)
pred = rnd_model.predict(x_test)
print('예측값 : ', [np.argmax(i) for i in pred[:3]])
print('실제값 : ', y_test[:3])

from sklearn.metrics import accuracy_score
print('RandomForestClassifier 정확도 : ', accuracy_score(onehot_test_labels, pred))  # 정확도:0.53

# 중요변수 확인
print('변수들 : ', x_data.columns)
print('특성(변수) 중요도 :\n{}'.format(rnd_model.feature_importances_))

def plot_feature_importances(model):  # 특성 중요도 시각화
    n_features = x_data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_data.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()

plot_feature_importances(rnd_model)  # average_monthly_hours, last_eval‎uation, satisfaction_level 이 중요변수

print("\n------ keras model ---------------------")
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x_data.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# 학습 도중 모델 저장
import os
model_dir = './tf10/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

modelpath = '{}salary.hdf5'.format(model_dir)

# print(modelpath)
chkpoint = ModelCheckpoint(filepath=modelpath, loss='loss', verbose=2, save_best_only=True)
es = EarlyStopping(monitor='loss', mode='auto', patience=10)

history = model.fit(x_train, onehot_train_labels, epochs=10000, batch_size = 256,
                    validation_split=0.2, verbose=2, shuffle=True, 
                    callbacks=[es, chkpoint])
print('eval:', model.evaluate(x_test, onehot_test_labels))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo--', label='train loss')
plt.plot(epochs, val_loss, 'r-', label='validation loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

pred = model.predict(x_test)
print(pred[0])
print(np.argmax(pred[0]))
print('결과 :', pre_y_data.categories[np.argmax(pred[0])])
