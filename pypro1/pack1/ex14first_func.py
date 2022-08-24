# 일급 함수 : 함수 안에 함수 선언, 인자로 함수 전달, 반환값으로 함수 사용

def func1(a, b):
    return a + b 

func2 = func1
print(func1(3,4))
print(func2(3,4))

def func3(func):    # 인자로 함수 전달
    def func4():    # 함수 안에 함수 선언
        print('나는 내부 함수라고 해~')
    func4()
    return func     # 반환값으로 함수 사용

mbc = func3(func1)
print(mbc(3, 4))

print('축약함수(Lambda - 이름이 없는 한 줄 짜리 함수)')
# 형식 : Lambda 인자,...:표현식 (단순한 함수일 때만 사용)

def Hap(x, y):
    return x + y

print(Hap(1,2))

print((lambda x, y:x + y)(1, 2))

g = lambda x, y:x * y
print(g)
print(g(3, 4))

print()
# lambda도 가변인수 사용 가능
kbs = lambda a, su=10: a + su
print(kbs(5))
print(kbs(5, 6)) # su = 6

sbs = lambda a, *tu, **di:print(a, tu, di)
sbs(5, 7, 8, m=4, n=5)

print()
# List에 람다를 넣어 사용
li = [lambda a, b:a+b, lambda a, b:a*b]
print(li[0](3, 4))
print(li[1](3, 4))

print('람다 적용해 보기 : 다른 함수에서 람다를 속성으로 사용')
# filter(function, iterable)
print(list(filter(lambda a:a < 5, range(10))))
print(list(filter(lambda a:a % 2, range(10))))

# 1 ~ 100 사이의 정수 중 5의 배수이거나 7의 배수만 출력하기 : filter
tmp = list(filter(lambda a: a % 5 == 0 or a % 7 == 0, range(1, 101)))
print(tmp)
