# 모듈(Module) : 소스 코드의 재사용을 가능하게 함
# 소스 코드를 하나의 이름 공간으로 구분하고 관리함
# 하나의 파일은 하나의 모듈이다.
# 표준 모듈, 사용자 정의 모듈, 제3자(Third party) 모듈(다운로드 받아서 불러다 씀)

# 내장된 모듈(표준 모듈) 읽어 사용하기 
a = 10
print(a)

import sys
print('모듈 경로 :', sys.path)
# sys.exit()  # 프로그램의 강제 종료 
print('종료')

import math
print(math.pi)
print(math.sin(math.radians(30)))

import calendar
calendar.setfirstweekday(6) # 일요일을 첫번째 요일로 셋팅
calendar.prmonth(2022, 7) # xxxx년 x월의 달력을 보여줌

print(dir(calendar)) # dir은 내장함수

# ...

print('난수 출력')
import random
print(random.random())  # 0 ~ 1 사이의 실수 출력
print(random.randrange(1, 10, 1))

from random import randrange, randint
print(randrange(1, 10, 1))  # 출처가 정확하지 않지만 모듈의 이름을 쓰지 않기 때문에 편할 수 있음
print(randint(1, 10))

from random import *    # 모두 로딩 (권장사항 아님)
print(randint(1, 10))
