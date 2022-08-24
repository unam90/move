# 1) step1 : array 관련 문제
#  정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 각 행 단위로 합계, 최댓값을 구하시오.
# < 출력 결과 예시>
# 1행 합계   : 0.8621332497162859
# 1행 최댓값 : 0.3422690004932227
# 2행 합계   : -1.5039264306910727
# 2행 최댓값 : 0.44626169669315
# 3행 합계   : 2.2852559938172514
# 3행 최댓값 : 1.5507574553572447

import numpy as np
data = np.random.randn(5, 4)
print(data)
print(data.sum(axis=1))
print(data.max(axis=1))

print()

i = 1
for r in data:
    print(str(i) + '행 합계 : ', r.sum())
    print(str(i) + '행 최대값 : ', r.max())
    i += 1


# 2) step2 : indexing 관련문제
#  문2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오.
#    조건1> 36개의 셀에 1~36까지 정수 채우기
#    조건2> 2번째 행 전체 원소 출력하기 
#               출력 결과 : [ 7.   8.   9.  10.  11.  12.]
#    조건3> 5번째 열 전체 원소 출력하기
#               출력결과 : [ 5. 11. 17. 23. 29. 35.]
#    조건4> 15~29 까지 아래 처럼 출력하기
#               출력결과 : 
#               [[15.  16.  17.]
#               [21.  22.  23]
#               [27.  28.  29.]]
  
z = np.zeros((6,6))
print(z)
num = 0
for r in range(6):
    for c in range(6):
       num += 1
       z[r, c] = num 

print(z)
print(z[1,:])
print(z[:,4])
print(z[2:5, 2:5])

print()
# 문2-2) 6행 4열의 다차원 zero 행렬 객체를 생성한 후 아래와 같이 처리하시오.
#      조건1> 20~100 사이의 난수 정수를 6개 발생시켜 각 행의 시작열에 난수 정수를 저장하고, 두 번째 열부터는 1씩 증가시켜 원소 저장하기
#      조건2> 첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기


p = np.zeros((6,4))  # 6행 4열의 다차원 zero행렬 객체 생성
print(p)
ran = np.random.randint(20, 100, 6)  # 20 ~ 100사이의 난수 정수 6개 발생
ran = list(ran)
print(ran)  
for r in range(len(p)):    
    num = ran.pop(0)  
    for c in range(len(p[0])):
        p[r][c] = num
        num += 1

print(p)
print()
p[0][:] = 1000  # 첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기
p[-1][:] = 6000
print(p)

# 3) step3 : unifunc 관련문제
#   표준정규분포를 따르는 난수를 이용하여 4행 5열 구조의 다차원 배열을 생성한 후
#   아래와 같이 넘파이 내장함수(유니버설 함수)를 이용하여 기술통계량을 구하시오.
#   배열 요소의 누적합을 출력하시오.

