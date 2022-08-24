# 서울지역 cctv 설치 데이터로 numpy + pandas + matplotlib 연습
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family = 'malgun gothic')  
plt.rcParams['axes.unicode_minus'] = False


cctv_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/cctv_seoul.csv")
print(cctv_data.head(3))
print(cctv_data.info())

cctv_data.rename(columns={cctv_data.columns[0]:'구별'}, inplace=True)
del cctv_data['2013년도 이전']
del cctv_data['2014년']
del cctv_data['2015년']
del cctv_data['2016년']
print(cctv_data.head(3))
print()


pop_seoul = pd.read_excel('https://github.com/pykwon/python/blob/master/testdata_utf8/population_seoul.xls?raw=true',
                          header=2, usecols='B,D,G,J,N')
pop_seoul.rename(columns={pop_seoul.columns[0]:'구별',
                          pop_seoul.columns[1]:'인구수',
                          pop_seoul.columns[2]:'한국인',
                          pop_seoul.columns[3]:'외국인',
                          pop_seoul.columns[4]:'고령자'}, inplace=True)
print(pop_seoul.head(3))
print(pop_seoul['구별'].unique())
print(cctv_data['구별'].unique())
print(len(cctv_data['구별'].unique()))

print()
pop_seoul['외국인비율'] = pop_seoul['외국인']/pop_seoul['인구수'] * 100
pop_seoul['고령자비율'] = pop_seoul['고령자']/pop_seoul['인구수'] * 100
print(pop_seoul.head(3))

print()
data_result = pd.merge(cctv_data, pop_seoul, on='구별')  # 공통칼럼 '구별'로 merge
print(data_result.head(3))
print()
data_result.set_index('구별', inplace=True)
print(data_result.head(3))

# 시각화 
plt.figure()
data_result['소계'].plot(kind='barh', grid=True, figsize=(8,8))
plt.title('구별 cctv 설치 현황')
plt.show()

print()
data_result['cctv비율'] = data_result['소계'] / data_result['인구수'] * 100
print(data_result.head(3))
data_result['cctv비율'].sort_values().plot(kind='pie')
plt.show()






