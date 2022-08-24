# 이항분류 : 딥러닝의 노드(뉴런)은 Logistic Regression 알고리즘을 따름
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers
import numpy as np

x_data = np.array([-50, -40, -30, -20, -10, -5, 0, 1, 10, 20, 30, 40, 50])
y_data = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])

# 1) Sequential API 
model = Sequential()
model.add(Flatten())  # 생략가능, 차원축소
model.add(Dense(1, input_dim=1, activation='sigmoid'))  # input_shape=(1,) /  activation을 안써주면 linear
opti = optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_data, y_data, batch_size=1, epochs=10, shuffle=False, verbose=1)
loss = model.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('evaluate:', loss)
 
import matplotlib.pyplot as plt
plt.plot(x_data, model.predict(x_data), 'b--', x_data, y_data, 'k.')
plt.show()

# 새로운 값으로 예측
print(model.predict([1,2,3.1,4.5,-21,99]).flatten())
print(np.squeeze(np.where(model.predict([-1,2,3.1,4.5,-21,99]) > 0.5, 1, 0)))

print()
# 2) function API 
from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(1,))
outputs = Dense(1, activation='sigmoid')(inputs)
model2 = Model(inputs, outputs)

model2.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(x_data, y_data, batch_size=1, epochs=10, shuffle=False, verbose=1)
loss2 = model2.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('evaluate2:', loss2)
 
import matplotlib.pyplot as plt
plt.plot(x_data, model2.predict(x_data), 'b--', x_data, y_data, 'k.')
plt.show()
