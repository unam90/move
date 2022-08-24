# JSON data <-> dict type의 호환
import json

dict = {'name':'tom', 'age':33, 'score':['90', '80', '100']}  # python의 dict
print('dict:%s'%dict)
print('dict type:%s'%type(dict))
print()
print('---JSON encoding : dict, list, tuple등을 JSON 모양의 문자열로 변경---')
str_val = json.dumps(dict)  # dumps를 거치면 string이 됨
print('str_val:%s'%str_val)
print('str_val type:%s'%type(str_val))
print(str_val[0:10])  # 문자열 관련 슬라이싱 가능
# print(str_val['name'])  # dict에서 쓰는 명령은 불가능
print()
print('---JSON decoding : JSON 모양의 문자열을 dict로 변경---')
json_val = json.loads(str_val)
print('json_val:%s'%json_val)
print('json_val type:%s'%type(json_val))
# print(json_val[0:10])  # 문자열 관련 슬라이싱 불가능
print(json_val['name'])  # dict 명령 가능

print()
for k in json_val.keys():
    print(k)
    
print()
for k in json_val.values():
    print(k)