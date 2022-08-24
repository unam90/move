# 단순 서버 : 1회용
from socket import *

serversock = socket(AF_INET, SOCK_STREAM)   # socket(소켓종류, 소켓유형)
serversock.bind(('172.30.1.18', 8888))        # ip, port 번호 지정
serversock.listen(1)                        # TCP listener 설정
print('server start...')

conn, addr = serversock.accept()            # 클라이언트 연결 대기
print('client addr :', addr)
print('from client msg :', conn.recv(1024).decode())
conn.close()
serversock.close()




