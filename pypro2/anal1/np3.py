# 배열 연산 : 기본적인 수학함수는 배열에 요소별로 적용 

import numpy as np

x = np.array([[1,2], [3,4]], dtype=np.float32)
y = np.arange(5, 9).reshape((2,2))  # 2행 2열 (2차원)으로 구조 변경
y = y.astype(np.float32)
print(x, x.dtype)
print(y, y.dtype)

print()
print(x + y)  # 요소별 합
print(np.add(x, y))
print()
print(x - y)  # 요소별 차
print(np.subtract(x, y))
print()
print(x * y)  # 요소별 곱
print(np.multiply(x, y))
print()
print(x / y)  # 요소별 나누기
print(np.divide(x, y))
print()
print('행렬곱 : 내적')
v = np.array([9,10])  # vector(1차원 배열)
w = np.array([11,12])
print(v * w)  # 요소별 곱 : 9 * 11, 10* 12
print(v.dot(w))  # 1차원 벡터에 행렬곱(내적)을 하면 결과는 scala가 됨/ 9*11 + 10*12
print(np.dot(v,w))  # v[0] * w[0] + v[1] * w[1]

print()
print(x)  # 2차원
print(v)  # 1차원
print(x * v)  # 요소별 곱. 결과는 큰 차원을 따름
print(x.dot(v))  # 행렬곱(내적). 결과는 낮은 차원을 따름
# x[0,0] * v[0] + x[0,1] * v[1] = 1*9 + 2*10 = 29
# x[1,0] * v[0] + x[1,1] * v[1] = 3*9 + 4*10 = 67
print(np.dot(x,v))

print()
print(x)
print(y) 
print(x.dot(y))
# x[0,0] * y[0,0] + x[0,1] * y[1,0] = 1*5 + 2*7 = 19
# x[0,0] * y[0,1] + x[0,1] * y[1,1] = 1*6 + 2*8 = 22
# x[1,0] * y[0,0] + x[1,1] * y[1,0] = 3*5 + 4*7 = 43
# x[1,0] * y[0,1] + x[1,1] * y[1,1] = 3*6 + 4*8 = 50

print(np.dot(x,y))

print()
print(x)
print(np.sum(x))
# axis로 작업 방향을 결정할 수 있다. 
print(np.sum(x, axis=0))  # 열방향 연산 
print(np.sum(x, axis=1))  # 행방향 연산

print()
print(np.mean(x))
print(np.argmax(x))  # 최대값 인덱스 
print(np.cumsum(x))  # 누적합

print(x)
print(x.T)  # 행과 열의 위치를 바꿈(전치 Transpose)
print(x.transpose()) # 행과 열의 위치를 바꿈(전치 Transpose)
print(x.swapaxes(0,1)) # 행과 열의 위치를 바꿈

# Broadcasting 연산 : 크기가 다른 배열 간의 연산
# 작은 배열과 큰 배열이 연산할 경우 작은 배열이 큰 배열의 크기만큼 연산에 반복적으로 참여 
x = np.arange(1, 10).reshape(3, 3)  # 1~9까지 3행 3열
y = np.array([1,2,3])
z = np.empty_like(x)  # x행렬과 크기가 같은 배열을 만들어준다. 안에 있는 값들은 의미가 없다.
print(x)
print(y)
# print(z)

# x와 y간의 더하기 연산
for i in range(3):
    z[i] = x[i] + y
    
print(z)

kbs = x + y  # numpy가 자동으로 broadcasting연산을 해준다.
print(kbs)








