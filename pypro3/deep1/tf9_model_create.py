# 단순선형회귀모델 작성 : 생성방법 3가지 경험하기
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, initializers
import numpy as np

# 공부에 투자한 시간에 따른 성적 결과 예측
x_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)  # feature
y_data = np.array([11, 32, 53, 64, 70], dtype=np.float32)  # label
print(np.corrcoef(x_data, y_data))  # 상관관계 확인 : 0.9743547 이고 두 변수 간에는 인과관계가 있다고 가정


print('1) Sequential API 사용: 가장 일반적이고 단순한 방법 -------')
model = Sequential()
model.add(Dense(units=2, input_dim=1, activation='linear'))
model.add(Dense(units=1, activation='linear'))
print(model.summary())

opti = optimizers.Adam(learning_rate = 0.01)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])  
# mse(min-squared error) : 평균제곱오차(작을수록 좋다) / 추측값에 대한 정확성을 측정하는 방법

history = model.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)

loss_metrics = model.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)  
# evaluate : 모델에 대한 성능 확인 (test 데이터를 넣는다)

print('lss_metrics:', loss_metrics)

from sklearn.metrics import r2_score
print('설명력:', r2_score(y_data, model.predict(x_data)))
print('실제값:', y_data)
print('예측값:', model.predict(x_data).flatten())

new_data=[1.5, 2.3, 5.8]
print('새 점수 예측 결과:', model.predict(new_data).flatten())

"""
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.plot(x_data, model.predict(x_data), 'b', x_data, y_data, 'ko')
plt.xlabel('공부시간')
plt.ylabel('점수')
plt.show()

# 학습 도중에 발생된 변화량을 시각화
plt.plot(history.history['mse'], label='평균제곱오차')
# plt.plot(history.history['loss'])
plt.xlabel('학습횟수')
plt.ylabel('mse')
plt.show()
"""

print('2) function API 사용 : 유연한 구조. 입력 데이터로부터 여러층을 공유하거나 다양한 입출력 사용 가능')
from keras.layers import Input
from keras.models import Model

# 각 층을 일종의 함수로써 처리를 함. 설계부분이 방법1과 다름
inputs = Input(shape=(1,))  # 입력 크기를 지정 (tuple로 작성)
# outputs = Dense(1, activation='linear')(inputs)  # 이전 층 레이어를 다음층 함수의 입력으로 사용
output1 = Dense(2, activation='linear')(inputs)
outputs = Dense(1, activation='linear')(output1)  
model2 = Model(inputs, outputs)

# 이하는 방법1과 같음
print(model2.summary())

opti = optimizers.Adam(learning_rate = 0.01)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])  
# mse(min-squared error) : 평균제곱오차(작을수록 좋다) / 추측값에 대한 정확성을 측정하는 방법

history2 = model2.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)

loss_metrics = model2.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss_metrics:', loss_metrics)

from sklearn.metrics import r2_score
print('설명력:', r2_score(y_data, model2.predict(x_data)))
print('실제값:', y_data)
print('예측값:', model2.predict(x_data).flatten())

print('3) sub classing 사용 : 동적인 구조 처리가 자유롭다. 고난이도의 작업에서 활용성이 가장 높다.')
x_data = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)  # feature
y_data = np.array([11, 32, 53, 64, 70], dtype=np.float32)  # label

class MyModel(Model):
    def __init__(self):  # layer 생성 후 call 메소드에서 수행하려는 연산을 적음
        super(MyModel, self).__init__()
        self.d1 = Dense(2, activation='linear')
        self.d2 = Dense(1, activation='linear')
        
    def call(self, x):
        inputs = self.d1(x)
        return self.d2(inputs)
        
model3 = MyModel()

# 이하는 방법1과 같음
opti = optimizers.Adam(learning_rate = 0.01)
model3.compile(optimizer=opti, loss='mse', metrics=['mse'])  
# mse(min-squared error) : 평균제곱오차(작을수록 좋다) / 추측값에 대한 정확성을 측정하는 방법

history3 = model3.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)

loss_metrics = model3.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss_metrics:', loss_metrics)

print('설명력:', r2_score(y_data, model3.predict(x_data)))
print('실제값:', y_data)
print('예측값:', model3.predict(x_data).flatten())

print(model3.summary())

print('3) sub classing 사용2 : custom layer + subclassing')
from keras.layers import Layer  # custom layer(사용자 정의층): 여러 레이어를 하나로 묶은 레이어를 구현할 경우 

class Linear(Layer):
    def __init__(self, units=1):
        super(Linear, self).__init__()
        self.units = units
        
    def build(self, input_shape):  # 모델의 가중치와 관련된 내용을 기술
        self.w = self.add_weight(shape=(input_shape[-1], self.units), # 입력의 크기를 알지 못할 때는 -1을 준다.
                                 initializer='random_normal', trainable = True)  # 역전파 진행   
        self.b = self.add_weight(shape=(self.units), initializer='zeros', trainable = True)
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b  # y = xw + b
    
class MyMLP(Model):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.linear1 = Linear(2)
        self.linear2 = Linear(1)
        
    def call(self, inputs):
        x = self.linear1(inputs)
        return self.linear2(x)

model4 = MyMLP()

# 이하는 방법1과 같음
opti = optimizers.Adam(learning_rate = 0.01)
model4.compile(optimizer=opti, loss='mse', metrics=['mse'])  
# mse(min-squared error) : 평균제곱오차(작을수록 좋다) / 추측값에 대한 정확성을 측정하는 방법

history4 = model4.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)

loss_metrics = model4.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss_metrics:', loss_metrics)

print('설명력:', r2_score(y_data, model4.predict(x_data)))
print('실제값:', y_data)
print('예측값:', model4.predict(x_data).flatten())

print(model4.summary())        
        
           
        
        
        
        
        
    
    
        

