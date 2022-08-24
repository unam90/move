# 재귀 함수(Recursive function) : 함수가 자기 자신을 호출. 재귀 호출
from conda.common._logic import TRUE

def countDown(n):
    if n == 0:
        print('완료')
    else:
        print(n, end = ' ')
        countDown(n - 1)  # 재귀 호출
        
countDown(5)

print()
def totFunc(n):
    if n == 1:
        print('탈출')
        return True
    return n + totFunc(n - 1)

result = totFunc(10)
print('10까지의 합은 ', result)

print('factorial(계승) 5! <= 5 * 4 * 3 * 2 * 1')
def fact_func(a):
    if a == 1:
        return 1
    print(a)
    return a * fact_func(a - 1)
    
result = fact_func(5)
print('5! : ', result)
