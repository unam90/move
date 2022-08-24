# closure(클로저) : scope에 제약을 받지 않는 변수들을 포함하고 있는 코드 블록이다.
# 함수 내에 선언된 변수를 함수 밖에서 참조가 가능하도록 하는 기술
from sympy.physics.units import amount

def funcTimes(a, b):
    c = a * b
    #print(c)
    return c
    
print(funcTimes(2, 3))

kbs = funcTimes(2, 3)
print(kbs)

kbs = funcTimes     # 함수의 주소를 kbs가 받은 것, 함수의 주소도 치환이 가능하다. 
# del funcTimes     # 함수를 지우기
print(kbs)
print(kbs(2,3))     # print(funcTimes(2, 3))과 동일
print(id(kbs), id(funcTimes))

mbc = sbs = kbs
print(mbc(2,3))
print(sbs(2,3))
print(kbs(2,3))
del kbs
print(mbc(2,3))
# print(kbs(2,3))   # kbs를 지워서 없음

print('\n클로저를 사용하지 않은 경우---------')
def out():
    count = 0
    def inn():
        nonlocal count
        count += 1
        return count
    print(inn())
    
# print(count)
out()
out()
out()

print('\n클로저를 사용한 경우---------')
def outer():
    count = 0
    def inner():
        nonlocal count
        count += 1
        return count
    return inner # <== 요것이 클로저 : 내부함수의 주소를 반환

var1 = outer()
print(var1)
print(var1())
print(var1())
print(var1())

var2 = outer()
print(var2())
print(var2())

print(id(var1), id(var2))

print('클로저 써먹기-----------') # 클로저 : 지역변수 값을 함수 바깥에서 참조 가능하다.
# 수량 * 단가 * 세금을 출력하는 함수 작성
def outer2(tax):    # tax는 지역변수 
    def inner2(su, dan):
        amount = su * dan * tax
        return amount
    return inner2  # <== 요것이 클로저

# 1분기에는 금액에 대한 세금(tax)이 0.1이 부과 
q1 =  outer2(0.1)
result1 = q1(5, 50000)
print('result1: ', result1)

result2 = q1(2, 10000) 
print('result2: ', result2)

# 2분기에는 금액에 대한 세금(tax)이 0.05이 부과 
q2 =  outer2(0.05)
result3 = q2(5, 50000)
print('result3: ', result3)
