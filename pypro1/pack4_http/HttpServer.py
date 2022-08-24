# 웹 서버 서비스 구현
from http.server import HTTPServer, CGIHTTPRequestHandler

# CGI(Common Gateway Interface) : 웹서버와 외부 프로그램 사이에서 정보를 주고 받는 방법이나 규약
# 대화형 웹 페이지를 작성할 수 있게 된다.

class Handler(CGIHTTPRequestHandler):
    cgi_directories = ['/cgi-bin']

serv = HTTPServer(('127.0.0.1', 8888), Handler)

print('웹 서버 서비스 시작...')
serv.serve_forever()

    
    

