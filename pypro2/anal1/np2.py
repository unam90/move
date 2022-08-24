# numpy 배열은 c배열을 이용한 파이썬 객체 

import numpy as np
from pygments.lexers import lisp

ss = [1, 2.5, True, 'tom']
print(ss, type(ss))

# numpy의 배열로 변환 : 같은 type 자료로만 구성
ss2 = np.array(ss)
print(ss2, type(ss2))  # class 'numpy.ndarray'

# 메모리 비교(list vs ndarray)
li = list(range(1, 10))
print(li)
print(id(li), id(li[0]), id(li[1]))
print(li * 10)  # li 요소를 10번 반복하라는 의미
print('~~' * 10)

# li 요소와 각각 10을 곱한 결과를 얻고 싶다면 for
for i in li:
    print(i * 10, end=' ')
print()
print([i * 10 for i in li])

print('-----')
num_arr = np.array(li)
print(id(num_arr), id(num_arr[0]), id(num_arr[1]))
print(num_arr * 10)

print('--1차원 배열 : vector---------')
a = np.array([1.,2,3])  # 상위 type int -> float -> complex -> str
print(a)
print(type(a), a.dtype, a.shape, a.ndim, a.size)  # a.shape는 각 차원의 요소의 개수 / ndim는 차원의 크기 / size는 요소 개수 
print(a[0], a[1], a[2])
a[0] = 5
print(a)

print('--2차원 배열 : matrix---------')
b = np.array([[1,2,3],[4,5,6]])  # 2행 3열
print(b)
print(type(b), b.dtype, b.shape, b.ndim, b.size)   
print(b[0,0], b[0,1], b[1,0])
print(b[[0]])  # 0행 값
print(b[[0]].flatten())  # 차원 축소
print(b[[0]].ravel())    # 차원 축소

print()
c = np.zeros((2, 3))  # 2행 3열짜리 매트리스 0으로 채움
print(c)

d = np.ones((2, 3))  # 1로 채움
print(d)

e = np.full((2, 3), 7)  # 7로 채움 
print(e)

f = np.eye(3)  # 주대각이 1인 단위 행렬 
print(f)

print()
print(np.random.rand(5), np.mean(np.random.rand(5)))  # 균등분호(데이터 값들이 일정함)
print(np.random.randn(5), np.mean(np.random.randn(5)))  # 정규분포
print(np.random.normal(0, 1, (2,3)))  # 2차원 matrix 2행 3열

np.random.seed(0)
x1 = np.random.randint(10, size=6)  # 1차원 0이상 10미만
x2 = np.random.randint(10, size=(3,4))  # 2차원 3행 4열 
x3 = np.random.randint(10, size=(3,4,5))  # 3차원
print(x1, x1.ndim, x1.size)
print(x2, x2.ndim, x2.size)
print(x3, x3.ndim, x3.size) 

print('-------------')
a = np.array([1,2,3,4,5])
print(a[1])
print(a[1:5:2])  # 1~5까지 2씩 증가
print(a[1:])  # 1번째 이후 
print(a[-2:])  # -2번째까지 (뒤에서부터 2번째까지)

b = a  # 주소를 치환
b[0] = 77
print(a)
print(b)
del b  

c = np.copy(a)  # 복사본을 만듬
c[0] = 88
print(a)
print(c)
del c 
# print(c)

print()
a = np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12]])
print(a)       # matrix
print(a[0])    # [1 2 3 4] vector
print(a[0,0])  # 1 scalar
print(a[[0]])  # [[1 2 3 4]] matrix
print(a[1:, 0:2]) # 1행 이후 0열과 1열만 출력

# sub array(서브 배열 : 배열 안에 있는 일부 배열을 사용하기)
print()
print(a)

b = a[:2, 1:3]  # a의 0~1행의 1~2열까지
print(b)
print(b[0,0])  # b의 0행 0열
b[0,0] = 99
print(b)
print(a)


