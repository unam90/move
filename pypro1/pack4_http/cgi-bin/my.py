# 클라이언트가 전송한 값 처리 

import cgi

form = cgi.FieldStorage()
name = form['name'].value   # java에서는 request.getParameter("name")
nai = form['age'].value

print('Content-Type:text/html;charset=utf-8\n')
print('''
<html>
<body>
<h2>my 문서</h2>
이름은 {0}님, 나이는 {1}세
<br>
<a href='../index.html'>메인으로</a>
</body>
</html>
'''.format(name, nai))