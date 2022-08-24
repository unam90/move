# 연산자

v1 = 3
v1 = v2 = v3 = 5 # 치환
print(v1, v2, v3)
print('출력1', end=',') # end='' : print시 개행되지 않고 이어서 쓰기
print('출력2')

v1 = 1, 2, 3 # 치환, 여러 값 담기 가능 !
print(v1)

v1, v2 = 10, 20 # 한 번에 여러 변수 값 각자 담기 가능
print(v1, v2)
v2, v1 = v1, v2
print(v1, v2) # 두 개의 기억저장소 값 맞바꾸기 가능 (자바는 불가능)

print('값 할당 packing 연산 : *')
v1, *v2 = 1, 2, 3, 4, 5
print(v1) # 1
print(v2) # [2, 3, 4, 5]  나머지 값을 모두 가짐

*v1, v2 = 1, 2, 3, 4, 5
print(v1) # [1, 2, 3, 4]
print(v2) # 5

v1, *v2, v3 = 1, 2, 3, 4, 5
print(v1)
print(v2)
print(v3)

print('--------------------------')

print(5 + 3, 5 - 3, 5 * 3, 5 / 3, 5 // 3, 5 % 3)
# / : 실수나누기(몫, 나머지 모두 취함)  // : 정수나누기 (몫만 취함)  % : 나눈 나머지만 취함

print(divmod(5, 3)) # 몫과 나머지를 별도로 취함 (몫, 나머지)

a, b = divmod(5, 3) # 몫, 나머지 각자 변수에 담아서 출력도 가능
print(a)
print(b)

# 거듭제곱
print(5 ** 3) # 5*5*5

print('우선 순위 : ', 3 + 4 * 5, (3 + 4) * 5)
# 우선순위 : 소괄호 > 산술(*, /  >  +, -) > 관계연산자 > 논리연산자 > 치환

print(5 > 3, 5 == 3, 5 != 3)

print(5 > 3 and 4 < 3, 5 > 3 or 4 < 3, not (5 >= 3)) # and : 둘 다 참이어야 참, or : 둘 중 하나만 참이면 참

print()
print('한' + '국인' + ' ' + "파'이'팅" + ' ' + '파"이"팅') # 더하기 : 숫자, 문자열 모두 가능
print('한국인' * 10) # 문자열에 대한 곱하기는 결국 그만큼 더하는 것 !

print('누적')
a = 10
a = a + 1
a += 1
# a++ 는 증감연산자, 사용불가
++a # 부호와 관련된 기호, 증감연산자 아님 !
print('a : ', a)

print(a, a * -1, -a, --a, +a) # --면 다시 양수

print()
print('bool 처리 : ', True, False)
print(bool(True), bool(1), bool(-3.4), bool('a')) # 0 이외 어떤 값을 가지고 있으면 True !
print(bool(False), bool(0), bool(0.0), bool(''), bool([]), bool({}), bool(None)) # 0이거나 값이 비워져 있으면 False

print('aa\nbb')
print('aa\tbb')
print('aa\bbb') # \b : backspace 
print(r'c:\aa\nbc') # r(raw string)을 사용해서 escape 기능 해제 (경로 등 표기시, 개행 안되고 그대로 출력됨)

print('print 관련 서식')
print(format(1.5678, '10.3f')) # 총 10자리, 소수점밑 3자리
print ('{0:.3f}'.format(1.0/3)) # 소수 셋째자리까지 표시 !
print('나는 나이가 %d 이다.'%23) # %d : 정수
print('나는 나이가 %s 이다.'%'스물셋') # %s : 문자(String)
print('나는 나이가 %d 이고 이름은 %s이다.'%(23, '홍길동'))
print('나는 나이가 %s 이고 이름은 %s이다.'%(23, '홍길동')) # 숫자도 %s 매핑 가능
print('나는 키가 %f이고, 에너지가 %d%%.'%(177.7, 100)) # %f : 실수 , %d%% : 백분율 표시

print('이름은 {0}, 나이는 {1}'.format('한국인', 33)) # 인덱스 0번째, 1번째
print('이름은 {}, 나이는 {}'.format('신선해', 33)) # 순서 안주면 자동으로 순서대로 들어감
print('이름은 {1}, 나이는 {0}'.format(34, '강나루'))
print('이름은 {1}, 나이는 {0} 나이는 {0} 나이는 {0}'.format(34, '강나루'))



