# 강남구 도서관 정보 json 문서 읽기
import json
import urllib.request as req

url ="https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json"
plainText = req.urlopen(url).read().decode()
print(plainText, type(plainText))  # <class 'str'>

jsonData = json.loads(plainText)
print(jsonData, type(jsonData))  # <class 'dict'>

print()
print(jsonData['SeoulLibraryTime']['row'][0]['LBRRY_NAME'])

# get 함수
libData = jsonData.get('SeoulLibraryTime').get('row')
print(libData)

print()
datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    print(name + ' ' + tel)
    tmp = [name, tel]
    datas.append(tmp)
    
import pandas as pd
df = pd.DataFrame(datas, columns=['도서관명', '전화'])
print(df)