# 집합(묶음)형 자료형 : str, list, tuple, set, dict (str 외는 문자+숫자 다 가능)

print('-------------str--------------')
# str : 문자열 자료형, 
#       순서가 있음 : 인덱싱, 슬라이싱 가능, 수정 불가(int, float, complex, bool, str, tuple)
#                    인덱싱 : 0부터 출발, 거꾸로 오른쪽에서부터 0, -1, -2, ... 도 가능(rfind)

s = 'sequence'
print(len(s), s.count('e')) # 문자열의 길이, 해당 문자의 개수
print(s.find('e'), s.find('e', 3), s.rfind('e')) # s.find('e', 3) 인덱스 3번부터 'e' 를 찾기, 'e'는 인덱스 4에 위치 

# 다양한 문자열 관련 함수들 -> 검색을 통해 사용 !

ss = 'mbc'
print(ss, id(ss))
ss = 'abc'
print(ss, id(ss))
# ss가 mbc를 참조하다가 새로운 인스턴스인 abc의 주소를 덮어쓰기 한 것 (수정불가, m을 a로 바꾼 게 아님)

print()
print(s) # sequence

# 인덱싱 (한글자씩 얻는 것)
print(s[0], s[7]) # s의 n번째 값 / s[8] 은 없는 값 - IndexError: string index out of range
print(s[-1], s[-2]) # 뒤에서부터 셈

# 슬라이싱
print(s[0:5], s[:3], s[3:]) # 0이상 5미만(0,1,2,3,4) , 3미만(0,1,2) , 3이상(3,4,5,6,7)
print(s[-4:-1], s[-4:], s[1:8:2]) # s[1:8:2] : 초기치, 목적치, 증가치(step) / 증가치 1이 기본값, 2이면 하나걸러 하나씩

s = 'fre' + s[2:] # 슬라이싱으로 얻어온 문자열에 새로운 문자를 더해서 인스턴싱함, 그 주소값을 s에 대입
print(s)

sss = 'mbc sbs kbs'
# 문자열 분리
imsi = sss.split(' ') 
print(imsi)

# 문자열 결합
imsi2 = ','.join(imsi)
print(imsi2)

# 문자열 치환
aa = 'Life is too long'
bb = aa.replace('Life', 'Your leg')
print(bb)

print('-------------List--------------')
# List 형식 : list(), [요소...] / 순서O, 수정O, 중복O

a = [1, 2, 3]
print(a)

b = [10, a, 20.5, True, '문자열'] # list 안에 list 넣기 가능 (중복 list)
print(b, id(b))
print(b[0], b[1], b[1][2]) # b[1][2] b의 1번째 안의 2번째

# 인덱싱이 가능 => 슬라이싱도 가능 !
print(b[2:5]) # 2부터 5미만
print(b[-2::2]) # -2부터 끝까지 2 간격으로
print(b[1][:2]) # b의 1번째 [1, 2, 3]에서 2번째까지만

print()
# list 추가, 삽입, 삭제, 수정
family = ['엄마', '아빠', '나']
family.append('남동생') # list 추가
family.insert(0, '할머니') # list 삽입
family.extend(['삼촌', '고모']) # list 여러개 삽입 / iterable : 집합형 자료
family += ['아저씨', '이모']
family.remove('나') # 값에 의한 list 삭제
del family[0] # 순서에 의한 삭제

print(family, len(family))

print('엄마' in family) # True
print('나' in family) # False

aa = [1,2,3,1,2]
print(aa, id(aa))
aa[0] = 77      # 요소 값 수정 (수정해도 주소 같음)
print(aa, id(aa)) 

print('자료구조 관련 : LiFO') # Last in FIRST OUT(선출후입)
kbs = [1,2,3]
kbs.append(4)
print(kbs)
kbs.pop() # 제일 위에껄 꺼냄 : 제일 마지막에 들어간 애가 제일 먼저 나옴
print(kbs)
kbs.pop()
print(kbs)

print('자료구조 관련 : FiFO') # First In First Out(선출선입), 파이프구조
kbs = [1,2,3]
kbs.append(4)
print(kbs)
kbs.pop(0)
print(kbs)
kbs.pop(0)
print(kbs)

print('-------------Tuple--------------')
# 형식 tuple(), (요소...) / List와 유사 - 순서O, 중복O, 수정X(Read Only), 대신 검색 속도 빠름

# t = ('a', 'b', 'c', 'd')
t = 'a', 'b', 'c', 'd' # 괄호 안넣어줘도 tuple로 출력 가능
print(t, type(t), id(t), len(t), t.index('c'))
      
p = (1, 2, 3, 1, 2)
print(p, id(p), type(p))
# p[0] = 77  # err : 'tuple' object does not support item assignment


# 형변환 (cast연산)
p2 = list(p) # tuple을 list로 바꿈
print(p2, type(p2))
p3 = tuple(p2) # list를 다시 tuple로 바꿈 (검색할 때 더 빠른 걸로 바꿈)
print(p3, type(p3))

print('-------------Set--------------')
# 형식 set(), {요소...} / 순서X(인덱싱, 슬라이싱 불가), 수정O, *중복X* UNIQUE !!!

a = {1, 2, 3, 1, 3}
print(a, type(a), len(a)) # 중복 불가 ! {1, 2, 3} <class 'set'> 3

b = {3, 4}
print(a.union(b)) # 합집합
print(a.intersection(b)) # 교집합
print(a | b) # 합집합
print(a - b) # 차집합
print(a & b) # 교집합
# print(b[0]) # TypeError: 'set' object is not subscriptable 인덱싱 불가 !

# 인덱싱은 안되지만 값은 함수로 추가 가능
b.add(5)
b.update({6, 7})
b.update([8, 9])
b.update((10,11,12))

# 값에 의한 삭제(순서 없음!)
b.discard(8)
b.remove(9)
b.discard(8) # 없는 값을 지우려고 하면 그냥 스킵
# b.remove(9) # 없는 값을 지우려고 하면 에러
print(b)

li = [1,2,2,3,4,4,5,3]
# 중복을 허용하고 싶지 않을 때 set에 담았다가 빼기 !
print(li)
imsi = set(li)
li = list(imsi)
print(li)

print('-------------Dict--------------') # JSON 형태
# 사전형 ! 형식 dict(), {'key':value, ...} / 순서X, 수정O, key는 중복X (value는 가능)

mydic = dict(k1=1, k2='abc', k3=1.2)
print(mydic, type(mydic), id(mydic))

dic = {'파이썬':'뱀', '자바':'커피', '스프링':'용수철'}
# dict 추가
dic['여름'] = '장마철'
print(dic, type(dic), len(dic))

# dict 삭제 : key이름 적어주면 됨
del dic['여름']
# dic.clear() # 모든 값 삭제

# dict 수정
dic['파이썬'] = '만능 언어'
print(dic, type(dic), len(dic))

# key값, value값 각각 출력 가능 - 리턴 타입 list
print(dic.keys())
print(dic.values())

print(dic.get('파이썬')) # 해당 key의 value값 얻기

print('파이썬' in dic) # True
print('파이' in dic) # False