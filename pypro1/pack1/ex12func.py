# 함수 : 인수(argument)와 매개변수(parameter)의 매칭
# 매개변수 유형 : 위치 매개변수, 기본값 매개변수, 키워드 매개변수, 가변 매개변수

def showGugu(start, end=5):
    for dan in range(start, end + 1):
        print(str(dan) + '단 출력')
        
print('뭔가를 하다가...')        
showGugu(2, 3) # 위치 매개변수
print()
showGugu(3) # 기본값 매개변수 
print()
showGugu(start=2, end=3) # 키워드 매개변수
print()
showGugu(end=3, start=2) # 키워드 매개변수
print()
showGugu(2, end=3)
print()
# showGugu(end=3, 2)
# showGugu(start=2, 3) 마지막 값에 이름을 안주면 에러
print()

print('가변 인수 처리 ---')

def func1(*ar):
    print(ar)
    for i in ar:
        print('음식:' + i)
        
func1('비빔밥', '공기밥', '볶음밥', '주먹밥') # tuple

print()

def func2(a, *ar):
# def func2(*ar, a):  # missing 1 required keyword-only argument: 'a'
    print(a)
    for i in ar:
        print('배고프면 ' + i)
        
func2('비빔밥', '공기밥', '볶음밥', '주먹밥')

print()
def selectProcess(choice, *ar):
    if choice == 'sum':
        re = 0
        for i in ar:
            re += i
    elif choice == 'mul':
        re = 1
        for i in ar:
            re *= i
        
    return re    
        
print(selectProcess('sum', 1,2,3,4,5))
print(selectProcess('mul', 1,2,3,4,5))

print('parameter가 dict type ----')
def func3(w, h, **other):
    print('몸무게{}, 키{}'.format(w, h))
    print(other)
    
func3(65, 176, irum='신기해', nai=23)
func3(65, 176, irum='신선해')
# func3(65, 176, {'irum':'신선해'}) # error

print()
def func4(a,b,*v1, **v2):
    print(a, b)
    print(v1)
    print(v2)
    
func4(1,2)
func4(1,2,3,4,5)
func4(1,2,3,4,5, m=6, n=7)
