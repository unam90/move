# 모듈의 멤버 중 클래스를 이용해 객체지향(중심)적인 프로그래밍 가능
# 클래스는 새로운 이름 공간을 지원하는단위로 멤버 메소드와 멤버변수를 갖는다.
# 클래스 단위로 instance 공간을 확보한다.
# 접근지정자 없다.(모두 다 public, getter/setter 필요없음) 메소드 오버로딩 없음.

a = 10
print('모듈의 멤버 중 a는', a, type(a))

# class 선언하기
class TestClass:  # class header
    aa = 1  # 멤버 변수(전역)
    
    def __init__(self):  # 시스템에 의해서 자동으로 호출됨(call back)
        print('생성자')
        
    def printMsg(self):  # 메소드(전역)
        name = '홍길동'    # 지역변수
        print(name)
        print(self.aa)
    
    def __del__(self):   # 클래스가 종료되면서 자동으로 호출
        print('소멸자')    # 시스템에 의해서 자동으로 호출됨(call back) 여기까지 class body

print('원형 클래스의 주소:', id(TestClass))  # 원형(prototype)class는 프로그램 실행시 자동으로 객체화 됨
print(TestClass.aa)
# TestClass.printMsg(self)

print()
test=TestClass()    # 생성자 호출. TestClass type의 객체가 생성이 됨(객체변수)
print('TestClass type의 새로운 객체의 주소:', id(test))
print(test.aa)
test.printMsg()     # Bound method call : 자동으로 객체변수가 인수로 전달됨(괄호 안에 객체변수가 자동으로 들어가기 때문에 적어줄 필요 없음)
print('------------')
TestClass.printMsg(test)    # Unbound method call : 자동으로 객체변수가 인수로 전달되지 않으므로 객체변수를 적어줘서 주소를 넘겨줘야 함

print()
print(type('kbs'))
print(type(test))
print(isinstance(test, TestClass))  # test는 TestClass type인가? True
del test    # 객체의 주소를 지우기
# print(isinstance(test, TestClass))  # NameError: name 'test' is not defined
print('종료')
