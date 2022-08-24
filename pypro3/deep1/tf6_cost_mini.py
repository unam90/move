# 회귀분석
# cost function : 예측값과 실제값의 차이의 평균
# cost minimize : 예측값과 실제값의 차이의 평균을 최소화. weight를 갱신 <= 최소제곱법
# 인공신경망은 델타규칙(경사하강법)을 이용하여 w(weight)를 갱신한다. 역시 내부적으로는 최소제곱법을 사용.

# 모델의 정확도가 높을수록 비용함수 값은 낮아지며, 반대의 경우 높아진다.

import math
import numpy as np
real = [10, 9, 3, 2, 11]  # 실제값
# pred = [11, 5, 2, 4, 3]   # 모델을 통해 얻은 예측값이 차이가 큰 경우 cost가 크다 = 17.2
pred = [10, 8, 4, 3, 11]  # 모델을 통해 얻은 예측값이 차이가 작은 경우 cost가 작다 = 0.6

cost = 0

for i in range(5):
    cost += math.pow(pred[i] - real[i], 2)  # (예측값 - 실제값)의 제곱의 합을 전체 수로 나누기 
    print(cost)
print()
print(cost / len(pred)) 

print()
print('최소제곱법을 통해 cost를 최소화한 후 시각화로 확인')
import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3,4,5]  # feature
y = [2,4,6,8,10]  # label
b = 0  # bias는 편의상 간섭을 최소화하기 위해 0을 주기로 하자.

w_val = []  # 가중치(w)의 변화값
cost_val = []  # cost의 변화값

for i in range(-30, 50):
    feed_w = i * 0.1   # learning_rate = 0.1을 준 것
    # print(feed_w)
    hypothesis = tf.multiply(feed_w, x) + b  # y = wx + b
    # cost = (예측값 - 실제값)의 제곱의 합을 전체 수로 나누기 
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    cost_val.append(cost)
    w_val.append(feed_w)
    print(str(i) + ', cost:' + str(cost.numpy()) + ', weight:' + str(feed_w))

plt.plot(w_val, cost_val, 'o')
plt.xlabel('w')
plt.ylabel('cost')
plt.show()