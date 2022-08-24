# 스레드를 이용해 날짜 및 시간 출력 / 파이썬은 구조적으로 스레드와 잘 맞지 않음
import time
aa = time.localtime()   # 전역변수는 스레드 간의 공유자원이 됨
print('현재는 {0}년 {1}월 {2}일'.format(aa.tm_year, aa.tm_mon, aa.tm_mday))
print('{0}시 {1}분 {2}초'.format(aa.tm_hour, aa.tm_min, aa.tm_sec))
print('오늘의 요일은 %d'%(aa.tm_wday))    # 월요일 : 0, ...일요일 : 6
print('오늘은 몇번째 날 %d'%(aa.tm_yday))

import threading

def time_show():
    now = time.localtime()
    print('현재는 {0}년 {1}월 {2}일'.format(now.tm_year, now.tm_mon, now.tm_mday), end=' ')
    print('{0}시 {1}분 {2}초'.format(now.tm_hour, now.tm_min, now.tm_sec))

def run():
    while True:
        now2 = time.localtime()
        if now2.tm_min == 52:break  # 원하는 분에 도달했을 때 스레드 멈추기
        
        time_show()
        time.sleep(1)

th = threading.Thread(target=run)
th.start()

th.join()

print('프로그램 종료')
   
time_show()
