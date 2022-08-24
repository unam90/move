# 배열에서 조건 연산 where(조건, 참, 거짓)
import numpy as np
 
x = np.array([1,2,3])
y = np.array([4,5,6])
conditionData = np.array([True, False, True])
result = np.where(conditionData, x, y)  # 참이면 x, 거짓이면 y
print(result)

print()
aa = np.where(x >= 2)
print(aa) 
print(np.where(x >= 2, 'T', 'F'))
print(np.where(x >= 2, x, x + 100))

print()
bb = np.random.randn(4, 4)  # 표준정규분포(가우시안분포)를 따르는 난수 발생
print(bb)
print(np.where(bb > 0, 2, bb))  # 0보다 클 경우 2, 아니면 bb값

print()
print('----배열 결합----')
kbs = np.concatenate([x, y])  # 1차원끼리 결합 2차원끼리 결합
print(kbs)

print()
print('----배열 분할----')
x1, x2 = np.split(kbs, 2)
print(x1)
print(x2)

print()
a = np.arange(1, 17).reshape(4, 4)
print(a)       
x1, x2 = np.hsplit(a, 2)  # 좌우 분리
print(x1)
print(x2)
print()
x1, x2 = np.vsplit(a, 2)  # 상하 분리
print(x1)
print(x2)

# 표본 추출(sampling)
# 복원 / 비복원 추출

li = np.array([1,2,3,4,5,6,7])

print()
# 복원 추출
for _ in range(5):
    print(li[np.random.randint(0, len(li) -1)], end = ' ')

print()
import random
# 비복원 추출
print(random.sample(list(li), k=5))
print()
print(random.sample(range(1, 46), k=6))
    
print()
# 복원 추출
print(list(np.random.choice(range(1, 46), 6)))  
print(list(np.random.choice(range(1, 46), 6, replace=True)))  # 복원
# 비복원 추출
print(list(np.random.choice(range(1, 46), 6, replace=False)))  # 비복원

# 가중치를 부여한 random 추출
ar = 'air book cat d e f god'
ar = ar.split(' ')  # 공백을 기준으로 자르기
print(ar)
print(np.random.choice(ar, 3, p=[0.1,0.1,0.1,0.1,0.1,0.1,0.4]))  # 선택확률(확률값을 높게 준 값이 많이 나옴)


