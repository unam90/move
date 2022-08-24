# 단순 클라이언트 : 1회용

from socket import *

clientsock = socket(AF_INET, SOCK_STREAM)
clientsock.connect(('172.30.1.18', 8888))     # TCP 서버와 연결을 시작
clientsock.send('안녕 반가워'.encode(encoding='utf-8', errors='strict'))
clientsock.close()


