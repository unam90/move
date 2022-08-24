# 클래스의 상속 이해
# Person <- Employee
#        <- Worker <- Programmer

from anaconda_navigator import static

class Person:
    say = '난 사람이야~'
    nai = '20'
    __good = '체력을 다지자'  # 앞에 __를 붙이면 private member가 됨
    
    def __init__(self, nai):
        print('Person 생성자')
        self.nai = nai
    
    def printInfo(self):
        print('나이:{}, 이야기:{}'.format(self.nai, self.say))
    
    def hello(self):
        print('안녕')
        print('private 멤버 출력:', self.__good)
        
    @staticmethod
    def sbs(tel):   # self가 없음
        print('staticmethod: 클래스 소속 - 클래스 멤버와 상관없는 독립적 처리를 할 경우에 사용, 어디서든 부를 수 있음')
        
print(Person.say, Person.nai)
p=Person('22')
p.printInfo()
p.hello()

print('--------Employee--------')
class Employee(Person):     # 상속
    say = '일하는 동물'
    subject = '근로자'
    
    def __init__(self):
        print('Employee 생성자')
    
    def printInfo(self):
        print('Employee 클래스의 printInfo')
        
    def e_printInfo(self):
        self.printInfo()    # 현재 클래스를 먼저 뒤진 다음, 부모에게 가서 찾는다.
        super().printInfo() # 바로 부모로 가서 찾는다.

e=Employee()
print(e.say, e.subject)
# e.printInfo()
e.e_printInfo()

print('--------Worker--------')
class Worker(Person):
    def __init__(self, nai):
        print('Worker 생성자')
        super().__init__(nai)   # Bound method call
        
    def w_printInfo(self):
        super().printInfo()     # Person의 printInfo()를 찾는 것
        # printInfo()는 함수를 부르는 것
        # self.printInfo()는 Worker의 printInfo()를 찾는 것
        
w=Worker('30')
print(w.say, w.nai)     # worker에 say가 없어서 부모로 가서 찾음, nai는 30dmf wna
w.w_printInfo()
w.printInfo()

print('-----Programmer-----')
class Programmer(Worker):
    def __init__(self, nai):
        print('Programmer 생성자')
        # super().__init__(nai)    # Bound method call (Worker로 감) / self가 묵시적으로 담긴다.
        Worker.__init__(self, nai) # Unbound method call
        
    def w_printInfo(self):
        print('Programmer에서 부모 생성자 override')
        
    # def hello(self):
    # print('private 멤버 출력:', self.__good)
    
pr=Programmer('33')
print(pr.say, pr.nai)
pr.w_printInfo()    # Programmer에서 찾은 w_printInfo()
pr.printInfo()      # Person에서 찾은 printInfo()

print('--------------') 
print(type(1.2))
print(type(pr))
print(type(w))
print(Programmer.__bases__) # Programmer의 부모는 Worker
print(Worker.__bases__) # Worker의 부모는 Person
print(Person.__bases__) # Person의 부모는 Object

print()
# pr.hello() # AttributeError: 'Programmer' object has no attribute '_Programmer__good'  

pr.sbs('111-1111')
Person.sbs('222-2222')  # 이렇게 사용하는 것을 권장

     









