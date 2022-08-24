
'''
여러 줄
주석
'''
"""
여러 줄
주석
"""

# 한 줄 주석

print("표준 출력장치로 출력")
var1 = "안녕 파이썬"
print(var1)
var1 = 3; print(var1); # 같은 줄에서 2개의 state를 줄 때 세미콜론으로 표시 !
var1 = '변수 선언 시 타입을 적지 않음, 참조 데이터에 의해 타입이 결정됨'
print(var1)

print()
a = 10
b = 12.3
c = b
print(a, b, c) # 주소를 치환함 ! 값을 치환한게 아님
print(id(a), type(a)) # 정수는 int
print(id(b), type(b)) # 실수는 float
print(id(c), type(c)) # b와 c는 똑같은 인스턴스를 참조

print(a is b, a == b) # False False  // is는 주소를 비교, ==은 값을 비교
print(b is c, b == c) # True True

aa = [100]
bb = [100]
print(aa == bb, aa is bb) # 값은 같은데 주소는 다름 !
print(id(aa), id(bb))

print()
A = 1; a = 2; # 파이썬은 대소문자를 구분함
print(A, ' ', a)

# for = 1 # 키워드는 변수로 사용불가 (클래스도 마찬가지, 사용자 정의명으로 쓸 수 없다는 것)

import keyword

print('키워드 목록 : ', keyword.kwlist)

print('\n숫자 진법')
print(10, oct(10), hex(10), bin(10)) # 10진수, 8진수, 16진수(0-9, a-f), 2진수
print(10, 0o12, 0xa, 0b1010)

print('\n자료형 확인') # 전부 object임
print(3, type(3))
print(3.4, type(3.4))
print(3 + 4j, type(3 + 4j)) # (3+4j) <class 'complex'> 복소수(실수+허수)
print(True, type(True)) # True <class 'bool'>
print('3', type('3')) # 3 <class 'str'>

# 묶음형 자료 (집합형 자료)
print((3,), type((3,))) # (3,) <class 'tuple'>
print([3], type([3])) # [3] <class 'list'>
print({3}, type({3})) # {3} <class 'set'>
print({'key':3}, type({'key':3})) # {'key': 3} <class 'dict'> - json과 잘맞음




