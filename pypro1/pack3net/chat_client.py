# 채팅 클라이언트
import socket
import threading
import sys

def handle(socket): # 서버가 보내준 메세지 받아들이는 역할
    while True:
        data = socket.recv(1024) # 채팅 메세지를 받기
        if not data:continue # 데이터(메세지)가 없을 때 계속함
        print(data.decode('utf-8')) # 데이터(메세지)가 있으면 메세지 디코딩해서 출력
        
# 파이썬의 표준출력은 버퍼링이 됨
sys.stdout.flush() # buffer를 비우기

name = input('채팅 아이디 입력:')
cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cs.connect(('127.0.0.1', 5000)) # server에 연결
cs.send(name.encode('utf-8')) # id를 보내기

th = threading.Thread(target=handle, args=(cs,))
th.start() # handle 메소드 처리 시작

while True:
    msg = input() # 채팅 메세지 입력
    sys.stdout.flush()   # buffer 비워주기
    if not msg:continue  # 메세지가 없으면 안넘어감
    cs.send(msg.encode('utf-8'))  # 메세지가 있으면 인코딩해서 보내기

cs.close()
    