# 정규 표현식 : 매우 중요 ***
# 정규 표현식(regular expression) : 특정한 규칙을 가진 문자열의 집합을 표현하는 데 사용하는 형식 언어
import re

ss = "1234 56 abc가나다mbc\nabcABC_123556_6python is fun파이썬 만세"

# 패턴 찾기 : findall
print(re.findall('123', ss)) # ['123', '123'] , list타입 (중복허용)
print(re.findall(r'123\n', ss))
print(re.findall(r'가나', ss))
print(re.findall(r'[1,2,3]', ss)) # 1,2,3만 찾아줌 (한글자씩)
print(re.findall(r'[1-3]', ss)) # 1,2,3만 찾아줌 (한글자씩)
print(re.findall(r'[0-9]', ss))
print(re.findall(r'[0-9]+', ss))  # + : 0-9 중에서 한글자 이상 연속으로 이어진 것 / *, +, ?, {횟수} 붙이기 가능
print(re.findall(r'[0-9]*', ss))  # * : 0개 이상
print(re.findall(r'[0-9]?', ss))  # ? : 0 또는 1

print(re.findall(r'[a-z,A-B]', ss))  # a-z : 영문자만 한글자씩
print(re.findall(r'[a-z,A-B]+', ss))
print(re.findall(r'[^a-z,A-B]+', ss))  # ^ : 부정, 영문자만 빼고 !
print(re.findall(r'[^가-힣]+', ss))

print(re.findall(r'[0-9]{2,3}', ss))  # 2글자짜리, 3글자짜리

# compile : 패턴을 객체화
pa = re.compile('[0-5]+')
print(re.findall(pa, ss))

imsi = re.findall(pa, ss)
print(imsi[0])

print()
print(re.findall(r'.bc', ss))  # . : 아무 글자나 가능
print(re.findall(r'a..', ss))
print(re.findall(r'^1', ss))  # ^ : 첫글자가 1로 시작되면 나옴 / 대괄호 앞에 적으면 부정, 대괄호 없이 적으면 이 글자가 첫글자라는 것
print(re.findall(r'만세$', ss))  # $ : 마지막 글자가 만세이면 나옴

print()
print(re.findall(r'\d', ss))  # \d : 숫자만
print(re.findall(r'\d+', ss))
print(re.findall(r'\d{2}', ss))
print(re.findall(r'\D', ss))  # \d를 뺀 나머지(숫자를 뺀 나머지)

print(re.findall(r'\s', ss))  # 공백이나 탭
print(re.findall(r'\S', ss))  # \s를 뺀 나머지 (반대)

print(re.findall(r'\w+', ss))  # 숫자, 문자만
print(re.findall(r'\W+', ss))  # 숫자, 문자를 뺀 나머지 (반대), 여집합

print(re.findall(r'\\', ss))  # \ 문자 자체

print()
# 그룹화
m = re.match(r'[0-9]+', ss)
print(m.group())

print()
p = re.compile('the', re.IGNORECASE)  # flag 사용 - IGNORECASE : 대소문자 구분X
print(p.findall('The dog the dog'))  # ['The', 'the'] 대소문자 구분 없이 the 리턴

print()
ss = """My name is tom.
I am happy"""           # 여러 행에 걸친 문자를 얻으려면 주석처럼 """ 주면 됨 !
print(ss)

p = re.compile('^.+', re.MULTILINE)  # ^.+ : 첫번째 글자 아무거나, 어떤 글자든 여러개
                                    # flag 사용 - 여러 행을 넣고 싶을 때 MULTILINE
print(p.findall(ss))
imsi = p.findall(ss)
print(imsi[0])
print(imsi[1])








