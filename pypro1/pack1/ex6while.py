# 반복문 while

a = 1

while a <= 5:
    print(a, end=' ')
    a += 1
    
print()
i = 1
while i <= 3:
    j = 1
    while j <= 4:
        print('i:' + str(i) + ',j:' + str(j))
        j = j + 1
    i += 1
    
print('1~100 사이 정수 중 3의 배수의 합')
i = 1
hap = 0
while i <= 100:
    if i % 3 == 0:
        hap += i 
    i += 1

print('합은 ' + str(hap))

print()
colors=['빨강', '초록', '파랑'] #set 으로 바꾸면 순서가 없어서 에러가 남

a=0
while a < len(colors):
    print(colors[a], end=' ')
    a += 1

print()
while colors:
    print(colors.pop())

print(len(colors))

print()
i = 1
while i <= 10:
    j = 1
    re = ''
    while j <= i:
        re = re + '*'
        j += 1
    print(re)
    i += 1
    
print('----------------')
import time
#print(time.sleep(3))
#sw = input('폭탄 스위치를 누를까요?[y/n]')
sw = 'n'
if sw == 'Y' or sw == 'y':
    count = 5
    while 1 <= count:
        print('%d초 남았어요'%count)
        time.sleep(1)
        count -=1
    print('폭발!!!')
elif sw == 'N' or sw =='n':
    print('작업 취소')
else:
    print('y 또는 n을 누르시오')

print('-----continue, break-----')    
a = 0 

while a < 10:
    a += 1
    if a == 5:continue
    if a == 7:break
    print(a)
else:
    print('while 정상 처리')


print('while 수행 후 %d'%a)

print('컴이 가진 임의의 정수 맞추기 ---')
import random
# random.seed(1) # 난수표 1번에 있는 값이 나온다. (난수 안에서 난수를 발생시키면 경우의 수가 더 많아진다.)
# print(random.random())  # 0 ~ 1 사이의 실수로 난수를 발생시킴
# print(random.randint(1, 10)) # 랜덤 정수 발생시키기
# friend = ['tom', 'james', 'oscar']
# print(random.choice(friend))
# print(random.sample(friend, 2))
# random.shuffle(friend)  # in-place 연산
# print(friend)

num = random.randint(1, 10)

while True:
    print('1 ~ 10 사이의 컴이 가진 예상 숫자를 입력:')
    guess = input()
    su = int(guess)
    if su == num:
        print('와우 성공~~~ ' * 5)
        break
    elif su < num:
        print('더 큰수를 입력해')
    elif su > num:
        print('더 작은수를 생각해')


print('\n종료')
  