# 기상청 제공 날씨정보 XML문서 처리
import urllib.request
import xml.etree.ElementTree as etree

try:
    webdata = urllib.request.urlopen("http://www.kma.go.kr/XML/weather/sfc_web_map.xml")
    webxml = webdata.read()
    webxml = webxml.decode('utf-8')
    print(webxml)
    webdata.close()
    with open('weather_xml.xml', mode='w', encoding='utf-8') as f:
        f.write(webxml)
except Exception as e:
    print('err: ', e)

xmlfile = etree.parse("weather_xml.xml")

root = xmlfile.getroot()
print(root.tag) # {current}current
print(root[0].tag) # {current}weather

children = root.findall('{current}weather')
# children = root.findall(root[0].tag)
print(children)

for it in children:
    y = it.get('year')
    m = it.get('month')
    d = it.get('day')
    h = it.get('hour')
    print(y + '년 ' + m + '월 ' + d + '일 ' + h + '시 현재')

datas = []
for child in root:
    for it in child:
        # print(it.tag)
        local_name = it.text  # 지역명
        temp = it.get('ta') # 온도
        desc = it.get('desc') # 상태
        # print(local_name, temp, desc)
        datas += [[local_name, temp, desc]]
        
print(datas)

import pandas as pd
df = pd.DataFrame(datas, columns=['지역', '온도', '상태'])
print(df.head(3))
print(df.tail(3))
print(len(df))

import numpy as np
imsi = np.array(df.온도, np.float32)
# print(imsi)
print('평균온도: \n', round(np.mean(imsi), 2))








