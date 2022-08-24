# 변수의 생존 범위 (scope rule)
# 접근순서 Local > Enclosing function > Global 

player = '전국대표'  # 전역변수 (모듈:module 의 어디서든 호출 가능)

def funcSoccer():
    name = '미스터 손'  # 지역변수 (현재 함수 내에서만 유효)
    player = '지역대표'
    print(name, player)
    
print(funcSoccer)
funcSoccer()
# print(name)  # NameError: name 'name' is not defined

print()
a = 10; b = 20; c = 30
print('1) a:{}, b:{}, c:{}'.format(a, b, c))

def func():
    a = 40
    b = 50
    c = 7
    
    def inner_func():
        # c = 60
        global c    # 전역변수화
        nonlocal b  # 부모함수의 변수 수준
        print('2) a:{}, b:{}, c:{}'.format(a, b, c))
        c = 60
        # UnboundLocalError: local variable 'c' referenced before assignment
        b = 70
        
        
    inner_func()
    print('3) a:{}, b:{}, c:{}'.format(a, b, c))

func()
print('4) a:{}, b:{}, c:{}'.format(a, b, c))
