# 자원의 재활용을 목적으로 클래스가 또 다른 클래스를 사용할 수 있다.
# 클래스의 포함 관계(has a 관계)
# 포함이란 어떤 객체가 다른 객체를 자신의 멤버 변수처럼 만들어 포함하고 있는 관계를 말한다.

# 핸들 클래스를 만들어 완성제품의 부품으로 활용
class PohamHandle:
    quantity = 0    # 회전량
    
    def leftTurn(self, quantity):
        self.quantity = quantity
        return '좌회전 '
    
    def rightTurn(self, quantity):
        self.quantity = quantity
        return '우회전 '
    
# 원래는 핸들을 별도의 파일로 만들고 완성차 클래스 만들 때 호출해야 하나, 
# 편의상 아래에 완성차 클래스를 작성하기로 하자.

class PohamCar:
    turnShow_msg = '정지 '
    
    def __init__(self, ownerName):
        self.ownerName = ownerName
        self.handle = PohamHandle()   # 클래스의 포함관계
    
    def playTurnHandle(self, q):
        if q > 0:
            self.turnShow_msg = self.handle.rightTurn(q)
        elif q < 0:
            self.turnShow_msg = self.handle.leftTurn(q)
        elif q == 0:
            self.turnShow_msg = "직진 "
            self.handle.quantity = 0
            
            
if __name__ == '__main__':
    tom = PohamCar('톰')
    tom.playTurnHandle(10)
    print(tom.ownerName + '의 회전량은 ' + tom.turnShow_msg + str(tom.handle.quantity))
    
    tom.playTurnHandle(0)
    print(tom.ownerName + '의 회전량은 ' + tom.turnShow_msg + str(tom.handle.quantity))
    
    print()
    james = PohamCar('제임스')
    tom.playTurnHandle(-5)
    print(james.ownerName + '의 회전량은 ' + james.turnShow_msg + str(james.handle.quantity))
    

