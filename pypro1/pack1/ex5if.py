# 조건 판단문 if
var = 6

if var >= 3:
    # print('크구나')
    # print('참이네')
    pass
else:
    print('거짓이구만')

print()
money = 100
age = 23

if money >= 500:
    item = 'apple'
    if age <= 30:
        msg = 'young'
    else:
        msg = 'old'
else:
    item = '바나나'
    if age >= 20:
        msg = ''
    else:
        msg = '어린이'

print(item, msg)

print()
# jum = int(input('점수 입력:')) 형변환 함수 int('123') 문자가 숫자가 됨, str(123) 숫자가 문자가 됨
# jum = jum + 1 형변환을 안하면 타입이 안맞아서 오류가 뜸
# print(str(jum) + '점') # 문자타입

jum = 70
res = ''
if jum >= 90:
    res = 'a'
elif jum >= 70:
    res = 'b'
else:
    res = 'c'
print('res : ', res)

if 90 <= jum <= 100:
    res = 'a'
elif 70 <= jum < 90:
    res = 'b'
else:
    res = 'c'
print('res : ', res, ' ', 'res:' + str(res))

print()
names = ['김혁규', '전해리', '최팀장']
if '전해리' in names: # not in (부정)
    print('친구야~~')
else:
    print('누구?')
    
print()
a = 'kbs'
b = 9 if a == 'kbs' else 11
print('b: ', b)

a = 11
b = 'mbc' if a == 9 else 'kbs'
print('b: ', b)    

a = 3
if a > 5:
    re = a * 2
else:
    re = a + 2
    
print('re:', re)

re = a * 2 if a > 5 else a + 2
print('re:', re)
a = 3
print((a+2, a*2)[a>5]) # 조건이 true면 1로 a*2를 수행함, false면 0으로 a+2를 수행함

a = 3
if a < 5:
    print(0)
elif a < 10:
    print(1)
else:
    print(2)
    
print(0 if a < 5 else 1 if a < 10 else 2)
print('종료')

