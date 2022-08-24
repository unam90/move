# 서버 서비스는 계속 유지

import socket
import sys

HOST = '127.0.0.1'
# HOST = ''    # 알아서 내 IP가 잡힘
PORT = 7878

serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

try:
    serversock.bind((HOST, PORT))
    serversock.listen(5)  # 동시 접속 최대 수 : 1 ~ 5
    print('서버 서비스 중 ...')
    
    while True:
        conn, addr = serversock.accept()
        print('클라이언트 정보 : ', addr[0], addr[1])
        print('수신 메세지:', conn.recv(1024).decode())
        
        # 서버가 클라이언트에게 메세지 전송
        conn.send(('from server:' + str(addr[0]) + ' 그대도 잘 지내시오').encode('utf-8'))
except Exception as e:
    print('err : ', e)
    sys.exit()
finally:
    serversock.close()
    


