# 다항분류
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np

x_data = np.array([[1,2,1,4],
                  [1,3,1,6],
                  [1,4,1,8],
                  [2,1,2,1],
                  [3,1,3,1],
                  [5,1,5,1],
                  [1,2,3,4],
                  [5,5,7,7]], dtype = np.float32)

# one hot vector로 label만들기
# 방법 1)
# y_data = np.array([[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]])
# 2,2,2,1,1,1,0,0

# 방법 2)
y_data = to_categorical([2,2,2,1,1,1,0,0])
print(y_data)

# model 생성
model = Sequential()
# model.add(Flatten())
# model.add(Dense(3, input_shape=(4,)))
# model.add(Activation('softmax'))

model.add(Dense(10, input_shape=(4,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

print(model.summary())

# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_data, y_data, epochs=100, verbose=0)

print('eval:', model.evaluate(x_data, y_data))

# predict
print(np.argmax(model.predict(np.array([[1,1,1,1]]))))  # 1
print(np.argmax(model.predict(np.array([[10,8,4,1]])))) # 0
