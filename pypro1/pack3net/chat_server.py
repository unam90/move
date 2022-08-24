# 멀티 채팅 서버 : socket, thread
import socket
import threading

ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.bind(('127.0.0.1', 5000))
ss.listen(5)
print('채팅 서버 서비스 시작...')
users = []

def chatUser(conn):
    name = conn.recv(1024)  # 접속 id(이름)를 받음
    data = '^^ ' + name.decode('utf-8') + '님 입장 ^^'
    print(data) # 서버 창에 찍고
    
    try:
        for p in users:
            p.send(data.encode('utf-8')) # 접속자들한테 누가 입장했다는 data 보내기
        
        while True:
            msg = conn.recv(1024) # 접속자들이 보낸 메세지를 받아서 
            suda_data = name.decode('utf-8') + '님 메세지:' + msg.decode('utf-8')   
            print(suda_data)   
            for p in users:
                p.send(suda_data.encode('utf-8')) # 모든 접속자들에게 메세지 전송 
    except:
        users.remove(conn) # 채팅방을 나간 경우
        data = '~~' + name.decode('utf-8') + '님 퇴장 ~~' 
        print(data)
        if users: # 접속자가 한명이라도 있으면 누가 채팅방 나갔다는 메세지 전송하기
            for p in users:
                p.send(data.encode('utf-8'))
        else: # 접속자가 없으면 끝내기
            print('exit')
    
while True:
    conn, addr = ss.accept()    # 채팅을 원하는 컴이 접속한 경우 실행
    users.append(conn)
    th = threading.Thread(target=chatUser, args=(conn,))
    th.start()