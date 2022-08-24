# 단순선형회귀 분석 모델(정량적(regression) 모델)을 작성 (feature : 연속형, label : 연속형)  - Keras 사용 2.x 
# 모델 학습방법 1) keras의 내장 API를 사용

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

x_data = [1.,2.,3.,4.,5.]
y_data = [1.2,2.0,3.0,3.5,5.5]
model=Sequential()   # 계층구조(Linear layer stack)를 이루는 모델을 정의
model.add(Dense(1, input_dim=1, activation='linear'))

# activation function의 종류 : https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations
sgd=optimizers.SGD(lr=0.01)  # 학습률(learning rate, lr)은 0.01

model.compile(optimizer=sgd, loss='mse',metrics=['mse'])  # 분류가 아니라 예측 모델이므로 mse
lossmetrics = model.evaluate(x_data,y_data)  # 학습 전 모델 성능 확인
print('lossmetrics:', lossmetrics)

# 옵티마이저는 경사하강법의 일종인 확률적 경사 하강법 sgd를 사용.
# 손실 함수(Loss function)은 평균제곱오차 mse를 사용.
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 100번 시도.

model.fit(x_data, y_data, batch_size=1, epochs=100, shuffle=False, verbose=0)

lossmetrics = model.evaluate(x_data,y_data)  # 학습 후 모델 성능 확인
print('lossmetrics:', lossmetrics)  # loss가 줄어듬

from sklearn.metrics import r2_score  # keras는 결정계수를 지원하지 않기때문에 sklearn을 통해 결정계수 확인
print('설명력 : ', r2_score(y_data, model.predict(x_data)))

print('예상 수 : ', model.predict([5]))       # [[4.801656]]
print('예상 수 : ', model.predict([2.5]))     # [[2.490468]]
print('예상 수 : ', model.predict([1.5, 3.3]))  # [[1.565993][3.230048]]

pred = model.predict(x_data)
import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'o')
plt.plot(x_data, pred, 'b-')
plt.xlabel('epochs')
plt.ylabel('y')
plt.show()

 