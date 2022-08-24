# 파이썬 모듈(파일)로 출력값을 웹브라우저로 전송
a = 10
b = 20
c = a + b
d = "결과는 " + str(c)

print('Content-Type:text/html;charset=utf-8\n')
print('<html><body>')
print('<b>안녕하세요</b> 파이썬 모듈로 작성한<br>문서입니다')
print('<hr> 파이썬 변수 값 출력 : %s'%(d,))
print('</body></html>')