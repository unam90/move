# local DB의 자료를 DataFrame 객체로 만들기 

import sqlite3

sql ="create table if not exists test(product varchar(10), maker varchar(10), weight real, price integer)"
conn = sqlite3.connect(':memory:')
conn.execute(sql)
conn.commit()

data1 = ('mouse', 'sm', 12.5, 5000)
data2 = [('keyboard', 'lg', 102.5, 25000),('monitor', 'lg', 1102.5, 525000)]
stmt = "insert into test values(?,?,?,?)"
conn.execute(stmt, data1)  # 데이터를 하나만 넣을 때
conn.executemany(stmt, data2)  # 여러개를 넣을 때
conn.commit()

cursor = conn.execute("select * from test")
rows = cursor.fetchall()
# print(rows)
for a in rows:
    print(a)

print()
print('-------DataFrame-------')
import pandas as pd
df1 = pd.DataFrame(rows, columns=['product', 'maker', 'weight', 'price'])
print(df1, type(df1))

print('-----------------------')
df2 = pd.read_sql("select * from test", conn)
print(df2)
# print(df2.to_html())

print('--------------------------------')
# DataFrame 자료를 DB로 저장
data = {
    'product':['연필','볼펜','노트'],
    'maker':['모나미','모나미','모나미'],
    'weight':[2.5, 3.4, 20.0],
    'price':[500,1000,5000]    
}
df3 = pd.DataFrame(data)
print(df3)
print('*********************************')
df3.to_sql('test', conn, if_exists='append', index=False)
df4 = pd.read_sql("select * from test", conn)
print(df4)

print()
print(pd.read_sql("select count(*) as 건수 from test", conn))






