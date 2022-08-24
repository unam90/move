# 웹문서 처리 : XML
# ElementTree 모듈 이용 
import xml.etree.ElementTree as etree

xmlfile = etree.parse("my.xml")  # 파싱
print(xmlfile, type(xmlfile))

root = xmlfile.getroot()
print(root.tag)       # items
print(root[0].tag)    # item
print(root[0][0].tag) # name
print(root[0][0].attrib) # {'id': 'ks1'}
print(root[0][2].attrib) # {'kor': '100', 'eng': '90'}
print(root[0][2].attrib.keys()) # dict_keys(['kor', 'eng'])
print(root[0][2].attrib.values()) # dict_values(['100', '90'])
imsi = list(root[0][2].attrib.values()) # ['100', '90']
print(imsi)

print('--------------')
children = root.findall('item')  # children은 item이 2개 
for it in children:
    re_id = it.find('name').get('id')  # 속성값의 id를 얻을 수 있음
    re_name = it.find('name').text  # 요소(element)값을 얻을 수 있음
    print(re_id, re_name)
    
    
    
    
    