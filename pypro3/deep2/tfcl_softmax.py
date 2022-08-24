import numpy as np
import matplotlib.pyplot as plt
 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
 
x = np.array([5, 9, 44])
y = softmax(x)
print(y)
print(np.argmax(y))
print(np.sum(y))

print('다항분류-------------------')
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical  # one-hot encoding을 지원
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
np.set_printoptions(suppress=True)  # 과학적 표기가 아닌 실수로 표현

# dataset
xdata = np.random.random((1000, 12))  # 1000행 12열 생성 / 시험점수
print(xdata[:3])
ydata = np.random.randint(5, size=(1000,1))  # 1000행 1열 / 카테고리(범주)는 5가지 / 국어:0 ~ 체육:4 라고 가정
print(ydata[:3])
ydata = to_categorical(ydata, num_classes=5)  # one-hot vector로 변환
print(ydata[:3])
print([np.argmax(i) for i in ydata[:3]])  # 다시 상수값으로 변환하여 확인

# model
model = Sequential()
model.add(Dense(units=512, input_shape=(12,), activation='relu'))  
# 뉴런의 갯수가 많아지면 학습량이 많아지고 모델의 성능이 좋아지지만 과적합의 문제가 있음
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=5, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(xdata, ydata, epochs=2000, batch_size=32, verbose=0)
model_eval = model.evaluate(xdata, ydata, batch_size=32, verbose=0)
print('평가결과:', model_eval)

# 시각화 
plt.plot(history.history['loss'],label='loss')
plt.xlabel('epochs')
plt.legend(loc=1)
plt.show()

plt.plot(history.history['accuracy'],label='accuracy')
plt.xlabel('epochs')
plt.legend(loc=2)
plt.show()

# 기존값으로 예측
print('예측값:', model.predict(xdata[:5]))
print('예측값:', [np.argmax(i) for i in model.predict(xdata[:5])])
print('실제값:', ydata[:5])
print('실제값:', [np.argmax(i) for i in ydata[:5]])

classes = np.array(['국어', '영어', '수학', '과학', '체육'])
print('예측값:', classes[np.argmax(model.predict(xdata[:5]), axis=-1)])

