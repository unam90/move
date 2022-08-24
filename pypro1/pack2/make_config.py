# DB 연결정보를 파일로 저장
import pickle

config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'maria123',
    'database':'test',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}

with open('mydb.dat', mode='wb') as obj:
    pickle.dump(config, obj)
