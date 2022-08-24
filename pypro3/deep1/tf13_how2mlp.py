# 회귀용 MLP 작성 : Sequential, Function api
# 캘리포니아 주택 가격 dataset 사용

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 표준화
from keras.models import Sequential
from keras.layers import Dense, Input, Concatenate
from keras import Model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = fetch_california_housing()
print(data.keys())
print(data.data[:3])
print(data.feature_names)
print(data.target[:3])
print(data.target_names)

# train / validation / test
print(data.data.shape)  # (20640, 8)
x_train_all, x_test, y_train_all, y_test = train_test_split(data.data, data.target, random_state=1)
print(x_train_all.shape, x_test.shape, y_train_all.shape, y_test.shape) 
# (15480, 8) (5160, 8) (15480,) (5160,)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=123)
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
# (11610, 8) (3870, 8) (11610,) (3870,)

# feature 특성들에 대해 표준화 작업
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.fit_transform(x_valid)
x_test = scaler.fit_transform(x_test)

print(x_test[:1])
print(scaler.inverse_transform(x_test[:1]))  # 표준화된 것을 원복하기

print('-----Sequential API로 단순한 형태의 MLP(deep learning network) 작성-----')
model = Sequential()
model.add(Dense(units=30, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(model.summary())

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid), verbose=2)
print('evaluate:', model.evaluate(x_test, y_test))

x_new = x_test[:3]
y_pred = model.predict(x_new)
print('예측값:', y_pred.ravel())  # ravel : 차원축소
print('실제값:', y_test[:3])

plt.plot(range(1, 11), history.history['mse'], c='b', label='mse')
plt.plot(range(1, 11), history.history['val_mse'], c='r', label='val_mse')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.legend()
plt.show()

print()
print('-----Function API로 복잡한 형태의 MLP(deep learning network)를 유연하게 작성-----')
input_ = Input(shape=x_train.shape[1:])  # 입력층
net1 = Dense(units=30, activation='relu')(input_)
net2 = Dense(units=30, activation='relu')(net1)  # 은닉층
concat = Concatenate()([input_, net2])  # 마지막 은닉층의 출력과 입력을 연결
output = Dense(units=1)(concat)  # 출력층

model2 = Model(inputs=[input_], outputs=[output])  # 최종적으로 입력과 출력을 지정하여 케라스 모델을 완성함

model2.compile(optimizer='adam', loss='mse', metrics=['mse'])

history = model2.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid), verbose=2)
print('evaluate:', model2.evaluate(x_test, y_test))

x_new = x_test[:3]
y_pred = model2.predict(x_new)
print('예측값:', y_pred.ravel())  # ravel : 차원축소
print('실제값:', y_test[:3])

plt.plot(range(1, 21), history.history['mse'], c='b', label='mse')
plt.plot(range(1, 21), history.history['val_mse'], c='r', label='val_mse')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.legend()
plt.show()


print()
print('--Function API : 일부 특성은 짧은 경로로 전달하고, 다른 특성들은 깊은 경로로 전달하는 모델 작성--')

# 5개의 특성(0~4)은 짧은 경로로, 6개의 특성(2~7)은 깊은 경로로 보낸다고 가정 
input_a = Input(shape=[5], name='wide_input')  # 층이 복잡할 때는 name을 주기
input_b = Input(shape=[6], name='deep_input')
net1 = Dense(units=30, activation='relu')(input_b)
net2 = Dense(units=30, activation='relu')(net1)
concat = Concatenate()([input_a, net2])
output = Dense(units=1, name='output')(concat)
model3 = Model(inputs=[input_a, input_b], outputs=[output])

model3.compile(optimizer='adam', loss='mse', metrics=['mse'])

# 입력 데이터 모양 feature가 커서 데이터 나누기
x_train_a, x_train_b = x_train[:, :5], x_train[:, 2:]
# print(x_train[:2])
# print(x_train_a[:2])
# print(x_train_b[:2])
x_valid_a, x_valid_b = x_valid[:, :5], x_valid[:, 2:]
x_test_a, x_test_b = x_test[:, :5], x_test[:, 2:]  # evaluate용
x_new_a, x_new_b = x_test_a[:3], x_test_b[:3]  # predict용

history = model3.fit((x_train_a, x_train_b), y_train, epochs=20, 
                     validation_data=((x_valid_a, x_valid_b), y_valid), verbose=2)
print('evaluate:', model3.evaluate((x_test_a, x_test_b), y_test))

y_pred3 = model3.predict((x_new_a, x_new_b))
print('예측값:', y_pred3.ravel())  # ravel : 차원축소
print('실제값:', y_test[:3])

plt.plot(range(1, 21), history.history['mse'], c='b', label='mse')
plt.plot(range(1, 21), history.history['val_mse'], c='r', label='val_mse')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.legend()
plt.show()