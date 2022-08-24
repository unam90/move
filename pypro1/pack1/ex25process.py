# 별도의 모듈로 작성된 클래스 호출하기

import pack1.ex25class_singer

def process():
    blackpink = pack1.ex25class_singer.Singer()
    print('타이틀 송:', blackpink.title_song)
    blackpink.sing()
    

if __name__ == '__main__':
    process()
    