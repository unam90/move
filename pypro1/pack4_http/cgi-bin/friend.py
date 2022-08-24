# 클라이언트가 전송한 값 처리 

import cgi

form = cgi.FieldStorage()
name = form['name'].value   # java에서는 request.getParameter("name")
phone = form['phone'].value
gen = form['gen'].value

print('Content-Type:text/html;charset=utf-8\n')
print('''
<html>
<body>
<h2>friend 문서</h2>
이름은 {0}, 번호는 {1}, 성별은 {2}
<br>
<a href='../index.html'>메인으로</a>
</body>
</html>
'''.format(name, phone, gen))