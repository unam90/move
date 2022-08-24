# 다중 상속(부모가 여러개)

class Tiger:
    data = '호랑이 세계'
    
    def cry(self):
        print('호랑이: 어흥ㅠㅠ 휴 더워')
        
    def eat(self):
        print('맹수는 고기를 좋아함. 고기 먹은 후 아아를 마심')
        
class Lion:
    def cry(self):
        print('사자: 으르렁~ 겁나 덥구만')
        
    def hobby(self):
        print('백수의 왕은 낮잠을 즐김')
        
class Liger1(Tiger, Lion):  # 다중 상속은 순서가 중요
    pass

l1 = Liger1()
l1.cry()    # 동일한 method가 있을 때 먼저 적은 Tiger cry에 우선순위가 있다.
l1.eat()
l1.hobby()
print(l1.data)

def hobby():
    print('이건 함수라고 해~')

print('-----------------')
class Liger2(Lion, Tiger):
    data = '라이거 만세'
    
    def play(self):
        print('라이거 고유 메소드')
        
    def hobby(self):
        print('라이거는 프로그램 짜기가 취미')
        
    def showData(self):
        self.hobby()     # Liger2의 hobby
        super().hobby()  # 부모로 바로 가서 찾음
        hobby()          # 그냥 함수
        self.eat()       # Liger2에서 먼저 eat을 찾고 없으면 부모에게서 찾음
        super().eat()    # 부모로 바로 가서 찾음
        print(self.data + ' ' + super().data)    # Liger2의 data와 Tiger의 data 
        self.play()
        
l2 = Liger2()
l2.play()
l2.showData()        
 