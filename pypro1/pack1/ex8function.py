# 함수 : 여러개의 수행문을 하나의 이름으로 묶어 놓은 실행 단위
# 독립적으로 구성된 프로그램 코드의 집합
# 반복 코드를 줄여주며, 짧은 시간에 효과적인 프로그래밍 가능 
# 내장함수, 사용자 정의함수 
# 사용자 정의함수 형식 
# def 함수명(argument,...):
#     함수 내용 ...

# 내장함수 일부 체험하기 
print(sum({1,2,3}))
print(bin(8))
print(int(1.7), float(3))
a = 10
b = eval('a + 5')
print(b)

print(round(1.2), round(1.6))

import math
print(math.ceil(1.2), math.ceil(1.6)) # 정수 근사치 중 큰 수 
print(math.floor(1.2), math.floor(1.6)) # 정수 근사치 중 작은 수 

print()
b_list = [True, 1, False]
print(all(b_list)) # 모두 참이면 참
print(any(b_list)) # 하나라도 참이면 참

b_list2 = [1,3,2,5,7,6]
result = all(a < 10 for a in b_list2)
print('모든 숫자가 10 미만이냐?', result)
result = any(a < 13 for a in b_list2)
print('모든 숫자 중 3미만이 있냐?', result)

print('복수 개의 집합 자료로 tuple 작성')
x = [10, 20, 30]
y = ['a', 'b']
for i in zip(x, y):
    print(i)
    
#...
