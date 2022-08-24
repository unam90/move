# Thread : Lightweight process 라고도 한다. 작업 실행단위를 말한다.
# process 내에서 여러개의 스레드를 운영해 멀티태스킹을 할 수 있다. 메모리를 공유.

import threading, time

def myrun(id):
    for i in range(1, 11):
        print('id={}-->{}'.format(id, i))
        time.sleep(0.3)
        
# 스레드를 사용하지 않은 경우
# myrun(1)
# myrun(2)

# 스레드를 사용한 경우
# threading.Thread(target='수행함수명')
th1 = threading.Thread(target=myrun, args=('일'))
th2 = threading.Thread(target=myrun, args=('이'))
th1.start()
th2.start()

th1.join()
th2.join()

print('프로그램 종료')
    
 