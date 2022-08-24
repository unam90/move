# 클래스의 이해

kor = 100       # 모듈의 멤버인 전역변수 

def abc():
    kor = 50    # 함수에서 유효한 지역변수
    print('모듈의 멤버인 함수')
    
class My:       # 모듈의 멤버인 클래스 
    # 생성자는 생략
    kor = 10    # 클래스의 멤버인 전역변수
    
    def abc(self):
        print('클래스의 멤버인 메소드')
    
    def showData(self):
        # kor = 30              # 메소드에서 유효한 지역변수 
        print('kor :', kor)     # 지역변수가 없으면 모듈의 멤버를 찾아간다.
        print('kor :', self.kor)
        print()
        self.abc()
        abc()
        
m = My()
m.showData()

print('----------------')
class Our:
    a = 1
    
print(Our.a)

our1 = Our()
print(our1.a)
our1.a = 2
print(our1.a)

our2 = Our()
print(our2.a)

print(Our.a)
