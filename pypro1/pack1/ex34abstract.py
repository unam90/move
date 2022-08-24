# 추상 클래스 - 추상 메소드
# 하위 클래스에서 부모의 메소드를 반드시 오버라이드 하도록 강요가 목적
# 추상 = 다형성, 강요

from abc import *

class AbstractClass(metaclass=ABCMeta):  # 추상 클래스 : 추상 클래스는 객체로 만들어질 수 없다. 하지만 생성자는 만들 수 있다.
    
    @abstractmethod
    def aaMethod(self):  # 추상 메소드
        pass
    
    def normalMethod(self):  # 일반 메소드
        print('추상 클래스 내의 일반 메소드')
        
# p = AbstractClass()  # TypeError: Can't instantiate abstract class
 
class Child1(AbstractClass):  # Child1 추상 클래스가 됨
    name = '난 Child1'
    
    def aaMethod(self):  # 오버라이딩을 강요 당함
        print('추상 메소드를 일반 메소드로 재정의')
        
c1 = Child1()
print(c1.name)
c1.aaMethod()
c1.normalMethod()  

print('-------------------')
class Child2(AbstractClass):
    def aaMethod(self):  # 오버라이딩을 강요 당함
        print('Child2에서 추상 메소드를 일반 메소드로 재정의')
        a = 120
        print('a:', a-20)

    def normalMethod(self):  # 일반 메소드를 필요에 의해 선택적으로 재정의
        print('Child2에서 부모와 기능이 다른 일반 메소드로 변경함')
        
c2 = Child2()

kbs = c1
kbs.aaMethod()
kbs.normalMethod()
print()
kbs = c2
kbs.aaMethod()
kbs.normalMethod()


    