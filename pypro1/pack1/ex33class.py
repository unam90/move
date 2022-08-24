# 다중 상속 연습 : 카페 연습문제 3번
class Animal:
    def move(self):
        pass

class Dog(Animal):  # 단일상송
    name = '개'
    def move(self):
        print('댕댕이는 낮에 활기차게 돌아다님')
        
class Cat(Animal):
    name = '고양이'
    def move(self):
        print('냥냥이는 밤에 돌아다님')
        print('눈빛이 빛남')

class Wolf(Dog, Cat):  # 다중상속
    pass

class Fox(Cat, Dog):
    def move(self):
        print('여우처럼 민첩하게')
    def foxMethod(self):
        print('여우 고유 메소드')

dog = Dog()
print(dog.name)    
dog.move()
print()
cat = Cat()
print(cat.name)
cat.move()
print('-----------------')
wolf = Wolf()
fox = Fox()

ani = wolf
ani.move() 
print()
ani = Fox()
ani.move() 

print()
print('^^^'*10)
anis = [dog, cat, wolf, fox]

for a in anis:
    a.move()
    print()       
    
    
print()
print(Fox.__mro__)  # 다중상속에서 순서를 확인 (클래스 탐색 순서)
print(Wolf.__mro__)  


