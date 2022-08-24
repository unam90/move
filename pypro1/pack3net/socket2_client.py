# 단순 클라이언트

from socket import *

clientsock = socket(AF_INET, SOCK_STREAM)
clientsock.connect(('127.0.0.1', 7878)) # TCP 서버와 연결을 시작
clientsock.sendall('안녕 반가워'.encode(encoding='utf-8'))
re_msg = clientsock.recv(1024).decode()
print('서버가 보내 자료는 ' + re_msg)

clientsock.close()
