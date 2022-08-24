# 네트워크를 위한 통신 채널 : socket
# socket : TCP/IP protocol의 프로그래머 인터페이스 이다.
# 프로세스 간에 대화가 가능하도록 하는 통신방식으로 client/server 모델에 기초한다.

import socket

print(socket.getservbyname('http', 'tcp'))
print(socket.getservbyname('telnet', 'tcp'))
print(socket.getservbyname('ftp', 'tcp'))
print(socket.getservbyname('smtp', 'tcp'))
print(socket.getservbyname('pop3', 'tcp'))
print()
print(socket.getaddrinfo('www.naver.com', 8, proto=socket.SOL_TCP))

