# 다항회귀
# 지역별 인구증가율과 고령인구비율(통계청 시각화 자료에서 발췌) 데이터로 선형회귀분석 및 시각화
import matplotlib.pyplot as plt
import tensorflow as tf
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
import random

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
print(len(population_inc))
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# plt.plot(population_inc,population_old,'bo')
# plt.xlabel('지역별 인구증가율 (%)')
# plt.ylabel('고령인구비율 (%)')
# plt.show()

# 지역별 인구증가율과 고령인구비율 : 이상(극단)치 제거 - 세종시 데이터
population_inc = population_inc[:5] + population_inc[6:]  # 5번째는 제외
population_old = population_old[:5] + population_old[6:]
print(len(population_inc))

# plt.plot(population_inc,population_old,'bo')
# plt.xlabel('지역별 인구증가율 (%)')
# plt.ylabel('고령인구비율 (%)')
# plt.show()

import numpy as np
print(np.corrcoef(population_inc, population_old))  # -0.10

# 1) 최소제곱법으로 회귀선 구하기 
x_var = sum(population_inc) / len(population_inc)
y_var = sum(population_old) / len(population_old)

# a(slope), b(bias)
a = sum([(y - y_var) * (x - x_var) for y, x in list(zip(population_old, population_inc))])
a /= sum([(x - x_var) **2 for x in population_inc])  # 기울기값
b = y_var - a * x_var
print('a:', a, ', b:', b)  # a : 기울기, b : 절편
print('예측값: ', -0.355834147915461 * 0.3 + 15.669317743971302)
print('실제값: ', 12.27)

# 일차방정식(회귀식)에 의한 회귀선을 시각화 
line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a * line_x + b
plt.plot(population_inc,population_old,'bo')
plt.plot(line_x, line_y, 'r-')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()

print()
# 2) keras로 회귀선 구하기

a = tf.Variable(random.random())  # weight(slope)
b = tf.Variable(random.random())

# 잔차제곱의 평균을 반환하는 함수
def compute_cost():
    y_pred = a * population_inc + b
    cost = tf.reduce_mean((population_old - y_pred) ** 2)
    return cost 
    
optimizer = tf.keras.optimizers.Adam(learning_rate=0.07)  
for i in range(1, 1001):
    optimizer.minimize(compute_cost, var_list=[a, b])  # 잔차제곱의 평균을 최소화하기
    if i % 100 == 0:
        print(i, 'a:', a.numpy(), ', b:', b.numpy(), ', cost(loss):', compute_cost().numpy())
    
line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a * line_x + b

plt.plot(population_inc,population_old,'bo')
plt.plot(line_x, line_y, 'r-')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()   

print()
# 3) keras로 다항 회귀선 구하기, 비선형, 2차함수 회귀선 y = ax^2 + bx + c
a = tf.Variable(random.random())  # weight(slope, 기울기)
b = tf.Variable(random.random())
c = tf.Variable(random.random())

# 잔차제곱의 평균을 반환하는 함수
def compute_cost2():
    y_pred = a * population_inc * population_inc + b * population_inc + c
    cost = tf.reduce_mean((population_old - y_pred) ** 2)
    return cost 

optimizer = tf.keras.optimizers.Adam(learning_rate=0.07)  
for i in range(1, 1001):
    optimizer.minimize(compute_cost2, var_list=[a, b, c])  # 잔차제곱의 평균을 최소화하기
    if i % 100 == 0:
        print(i, 'a:', a.numpy(), ', b:', b.numpy(), ', c:', c.numpy(), ', cost(loss):', compute_cost2().numpy())
    
line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a * line_x * line_x + b * line_x + c  # ax^2 + bx + c

plt.plot(population_inc,population_old,'bo')
plt.plot(line_x, line_y, 'r-')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()   


print()
# 4) keras로 네트워크 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
    
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')
model.summary()

model.fit(population_inc, population_old, epochs=100)
print(model.predict(population_inc).flatten())

line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = model.predict(line_x)
plt.plot(population_inc,population_old,'bo')
plt.plot(line_x, line_y, 'r-')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()  