# 사용자 정의 모듈 작성 후 읽기
print('현재 모듈에서 뭔가를 하다가 다른 모듈 호출하기')
a = 10
print('a : ', a)
print(dir())


print()
import pack1.ex18mymod1

list1 = [1, 3]
list2 = [2, 4]
pack1.ex18mymod1.ListHap(list1, list2)

def abc():
    if __name__ == '__main__':
        print('여기가 최상위 모듈이라고 외칩니다')
        
abc()

pack1.ex18mymod1.Kbs()

from pack1 import ex18mymod1
ex18mymod1.Kbs()

from pack1.ex18mymod1 import Kbs, Mbc, name
Kbs()
Mbc()
print(name)

print()
# 다른 패키지의 모듈 읽기
from pack_other import ex18mymod2
ex18mymod2.Hap(5, 3)

import pack_other.ex18mymod2
pack_other.ex18mymod2.Cha(5, 3)

print()
# path가 설정된 지역의 모듈 호출 (c드라이브 anaconda3 lib폴더에 ex18momod3.py를 붙여넣음)
import ex18mymod3   # package를 import할 필요 없음
ex18mymod3.Gop(5, 3)

from ex18mymod3 import Nanugi
Nanugi(5, 3)

