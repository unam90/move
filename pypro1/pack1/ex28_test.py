# 클래스 포함관계 연습문제 : 커피 자판기 프로그램

class CoinIn:
    # coin = 0
    # change = 0
    
    def calc(self, cupCount):
        result = ''
        
        if self.coin < 200:
            result = '요금이 부족합니다.'
        elif cupCount * 200 > self.coin:
            result = '요금이 부족합니다.'
        else:
            self.change = self.coin - (cupCount * 200)
            result = '커피 ' + str(cupCount) + '잔과 잔돈 ' + str(self.change) +'원'
        
        return result
    
class Machine:
    def __init__(self):
        self.coinIn = CoinIn()  # Machine이 CoinIn을 포함하고 있음. 클래스의 포함
    
    def showData(self):
        self.coinIn.coin = int(input('동전을 입력하세요:'))
        self.cupCount = int(input('몇 잔을 원하세요:'))
        print(self.coinIn.calc(self.cupCount))
    
Machine().showData()
    