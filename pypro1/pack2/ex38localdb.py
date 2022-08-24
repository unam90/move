# 개인용 RDBMS - sqlite3 : 파이썬에 기본 모듈로 제공

import sqlite3
print(sqlite3.sqlite_version)

print()
# conn = sqlite3.connect('example.db') # 데이터베이스 생성
conn = sqlite3.connect(':memory:')     # 테스트용 - 주기억장치(RAM)에 저장됨(휘발성)

try:
    cursor = conn.cursor()  # 커서 객체로 SQL문 처리
    
    # table 작성    # sqlite3 data type : integer, real, text, blob...
    cursor.execute("create table if not exists friend(name text, phone text, addr text)")
    
    # insert data
    cursor.execute("insert into friend(name, phone, addr) values('한국인', '111-1111', '역삼1동')")
    cursor.execute("insert into friend values('신기해', '222-1111', '역삼2동')")
    input_data = ('조조', '333-1111', '서초2동')
    cursor.execute("insert into friend values(?,?,?)", input_data)
    conn.commit()
    
    # select data
    cursor.execute("select * from friend")
    # print(cursor.fetchone())
    print(cursor.fetchall())
    
    print()
    cursor.execute("select name, addr, phone from friend")
    for c in cursor:
        # print(c)
        print(c[0] + ' ' + c[1] + ' ' + c[2])
    
except Exception as e:
    print('err:', e)
    conn.rollback()
finally:
    conn.close()    # 메모리 해제 
