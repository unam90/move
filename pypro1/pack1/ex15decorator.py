# 함수 장식자 (function decorator) : Meta 기능이 있다.
# 함수 장식자는 또 다른 함수를 감싼 함수다.
# 장식자는 포장된 함수로 교체하여 함수를 반환한다.

def make2(fn):
    return lambda:'안녕 ' + fn()

def make1(fn):
    return lambda:'반가워 ' + fn()

def hello():
    return "홍길동"

hi = make2(make1(hello))
print(hi())

print()

@make2              # decorator
@make1              # hello2의 주소를 make1에게 주고 make1의 주소를 make2에게 줌
def hello2():       
    return "홍길자"

print(hello2())

print()
hi2 = hello2()  # 실행 결과 치환
print(hi2)
hi3 = hello2    # 함수 주소 치환
print(hi3())
