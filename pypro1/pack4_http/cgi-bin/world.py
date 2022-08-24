s1 = '자료1'
s2 = '두번째 자료'

print('Content-Type:text/html;charset=utf-8\n')
print('''
<html>
<head>
<meta charset="UTF-8">
<title>world</title>
</head>
<body>
<h2>world 문서</h2>
자료 출력 : {0}, {1}
<br>
<img src='../images/abc.png' width='60%' />
<br>
<a href='../index.html'>메인으로</a>
<br>
작성자 : 홍길동
</body>
</html>
'''.format(s1, s2))