# 상관분석 선행 연습 - 공분산/상관계수
import numpy as np

print('공분산:', np.cov(np.arange(1, 6), np.arange(2, 7)))
print('공분산:', np.cov(np.arange(10, 60, 10), np.arange(20, 70, 10)))
print('공분산:', np.cov(np.arange(1, 6), (3, 3, 3, 3, 3)))
print('공분산:', np.cov(np.arange(1, 6), np.arange(6, 1, -1)))

print()
x = [8,3,6,6,9,4,3,9,3,4]
print('x 평균:', np.mean(x), ', x 분산:', np.var(x))  # 분산 : 평균으로부터 데이터가 떨어진 정도

# y = [6,2,4,6,9,5,1,8,4,5]
y = [600,200,400,600,900,500,100,800,400,500]
print('y 평균:', np.mean(y), ', y 분산:', np.var(y))  # 분산 : 평균으로부터 데이터가 떨어진 정도

print('x, y의 공분산 : ', np.cov(x, y)[0,1])
print('x, y의 상관계수 : ', np.corrcoef(x, y)[0,1])

print()
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.show()

# 주의 : 공분산이나 상관계수는 선형적인 관계를 측정하기 때문에 비선형적인 데이터 분포를 가진 경우는 사용하지 않는다.
m = [-3, -2, -1, 0, 1, 2, 3]
n = [9, 4, 1, 0, 1, 4, 9]
print('공분산:', np.cov(m, n)[0, 1])  
print('상관계수:', np.corrcoef(m, n)[0, 1])

plt.scatter(m, n)
plt.show()