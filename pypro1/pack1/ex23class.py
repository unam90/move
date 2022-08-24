# 클래스의 이해
print('어쩌구 저쩌구 하다가 ...')

class Car:
    handle = 0      # handle과 speed는 Car의 멤버 변수 (공유멤버)
    speed = 0       
    
    def __init__(self, speed, name):    # speed, name은 지역변수
        self.speed = speed
        self.name = name
       
    def showData(self):    # 메소드
        km = '킬로미터'
        msg = '속도는 ' + str(self.speed) + km
        return msg + ', 핸들은 ' + str(self.handle)
    
print(Car.handle, Car.speed)
print()

car1 = Car(5, 'tom')   # Car type의 새로운 객체인 car1이 self로 들어감
print(car1.handle, car1.speed, car1.name)   # car1에는 handle이 없기 때문에 handle은 원형클래스의 멤버를 참조(지역이 우선)
car1.color = '파랑'     # color는 car1만 갖고 있는 것
print('car1 자동차 색은', car1.color)

car2 = Car(10, 'john')
print(car2.handle, car2.speed, car2.name)
# print('car2 자동차 색은', car2.color) # AttributeError: 'Car' object has no attribute 'color'

print()
print('method')
print('car1의', car1.showData())
print('car2의', car2.showData())
print()
car1.speed = 100
print('car1의', car1.showData())
print('car2의', car2.showData())
print('원형클래스의 속도는', Car.speed)

print()
Car.handle = 1
print('car1의', car1.showData())
print('car2의', car2.showData())

print()
print(id(Car), id(car1), id(car2))
print(type(car1), type(car2))
print()
print(car1.__dict__)    # 객체의 멤버 확인
print(car2.__dict__)

