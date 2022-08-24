# 카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
#   예제파일 : cleanDescriptive.csv
#   칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
#   조건 :  level, pass에 대해 NA가 있는 행은 제외한다.

# 귀무 : 부모학력 수준은 자녀의 진학여부와 관련이 없다.
# 대립 : 부모학력 수준은 자녀의 진학여부과 관련이 있다.
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('testdata/cleanDescriptive.csv')

print(data.head(3))
print(type(data))
print(data.level.unique())  # 부모의 학력 수준
print(data['pass'].unique())  # 자녀의 대학 진학 여부

ctab = pd.crosstab(index=data['level'], columns=data['pass'], dropna=True)
print(ctab)

chi2, p, ddof, exp = stats.chi2_contingency(ctab)
# print('chi2:{}, p-value:{}, df:{}'.format(chi2, p, ddof))
print('chi2:', chi2, ', p:', p)
# chi2:2.766951202595669, p-value:0.25070568406521354, df:2
# 해석:p-value는 0.2507 > 0.05 이므로 대립가설 기각, 귀무가설 채택
# 부모학력 수준은 자녀의 진학여부와 관련이 없다.

print()
print('-------------------------------------------')
# 카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
# 그렇다면 정말로 jikwon_jik과 jikwon_pay 간의 관련성이 없는지 분석. 가설검정하시오.
#   예제파일 : MariaDB의 jikwon table 
#   jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#   jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
#   조건 : NA가 있는 행은 제외한다.
# 귀무 : A회사의 직급과 연봉은 관련이 없다.
# 대립 : A회사의 직급과 연봉은 관련이 있다.

import MySQLdb
import numpy as np

config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'maria123',
    'database':'test',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}

try:
    conn = MySQLdb.connect(**config)    
    cursor = conn.cursor()     
    
    sql = "select jikwon_jik, jikwon_pay from jikwon"
    cursor.execute(sql)
     
    df = pd.DataFrame(cursor.fetchall(), columns=['직급', '연봉'])
    print(df.head(3))
    
    # jikwon_jik (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
    # 함수를 데이터 프레임에 적용하는 방법
    # 데이터프레임['칼럼명'].apply(함수명)
    df['직급'] = df['직급'].apply(
        lambda g:1 if g =='이사' else 2 if g =='부장' else 3 \
                    if g =='과장' else 4 if g =='대리' else 5)
    
    print(df)
    # jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
    # np.where 값 반환 응용 np.where(조건, True일 때, False일 때 위치 브로드캐스팅)
    # 첫 번째 인자의 조건을 만족하면 해당 위치는 두 번째 인자의 연산으로 브로드캐스팅되고, 
    # 만족하지 않는 경우는 세 번째 인자의 연산으로 브로드캐스팅된다.
    df['연봉'] = np.where(df['연봉'] < 3000, 1, df['연봉'])
    df['연봉'] = np.where((df['연봉'] >= 3000) & (df['연봉'] < 5000), 2, df['연봉'])
    df['연봉'] = np.where((df['연봉'] >= 5000) & (df['연봉'] < 7000), 3, df['연봉'])
    df['연봉'] = np.where(df['연봉'] >= 7000, 4, df['연봉'])
    
    print(df.head(5))
    print()
    ctab = pd.crosstab(index=df['직급'], columns=df['연봉'], dropna=True)  # NA가 있는 행은 제외
    print(ctab)
    
    chi2, p, ddof, exp = stats.chi2_contingency(ctab)
    print('chi2:', chi2, ', p:', p)  
    # chi2: 37.40349394195548 , p: 0.00019211533885350577
    # p-value : 0.000192 < 0.05 로 귀무가설 기각, 대립가설 채택
    # A회사의 직급과 연봉은 관련이 있다.

except Exception as e:
    print('에러:', e)
finally:
    cursor.close()
    conn.close()







