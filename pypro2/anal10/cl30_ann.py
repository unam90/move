# Neural Network 
# 사람이 뇌를 사용해 문제를 처리하는 방법과 비슷한 방법으로 문제를 해결하기 위한 알고리즘을 사용한다.
# 사람은 뇌의 기본 구조 조직인 뉴런(neuron)과 뉴런이 연결되어 일을 처리하는 것처럼, 수학적 모델로서의
# 뉴런이 상호 연결되어 네트워크를 형성할 때 이를 신경망이라 한다.

# Perceptron (단층 신경망 - 뉴런(node)을 1개 사용)
# input data(feature) * weight(가중치) + bias(편향값)의 합에 대해 
# 임계값을 기준으로 output(출력값)을 내보냄
# y = wx + b라고 하는 일차방정식을 사용
# 논리회로 분류를 위한 간단한 실습(이항분류)

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
print(feature)
# label = np.array([0,0,0,1]) # and
# label = np.array([0,1,1,1]) # or
label = np.array([0,1,1,0]) # xor

ml = Perceptron(max_iter=10000, eta0=0.1).fit(feature, label)  # max_iter=학습횟수, eta0=학습량
print(ml)
pred = ml.predict(feature)
print('pred:', pred)
print('acc:', accuracy_score(label, pred))


 