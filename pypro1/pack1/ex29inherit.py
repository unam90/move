# 자원의 재활용을 목적으로 클래스를 상속 가능. 다중 상속도 허용

class Animal:
    nai = 1
    
    def __init__(self):     # 클래스를 생성하면 자동으로 호출된다.
        print('Animal 생성자')
        
    def move(self):
        print('움직이는 생물')

class Dog(Animal):  # 상속
    erum = '난 댕댕이'
    
    def __init__(self):
        print('Dog 생성자')    # 부모, 자식 둘다 생성자가 있을 경우에 '자식 생성자만' 수행된다.
        
    def my(self):
        print(self.erum + '만세')

dog1 = Dog()
dog1.my()
dog1.move()
print('nai:',dog1.nai)

print()
class Horse(Animal):
    pass

horse1 = Horse()
horse1.move()
print('nai:',horse1.nai)



