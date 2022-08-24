# 카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
#   예제파일 : cleanDescriptive.csv
#   칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
#   조건 : NA가 있는 행은 제외한다.

# 귀무 : 부모학력 수준이 자녀의 진학여부와 관련이 없다.
# 대립 : 부모학력 수준이 자녀의 진학여부와 관련이 있다.

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('testdata/cleanDescriptive.csv')
print(data.head(5))

print(data.level.unique()) # 부모의 학력 수준
print(data['pass'].unique()) # 자녀의 대학 진학 여부

ctab = pd.crosstab(index=data['level'], columns=data['pass'], dropna=True)
print(ctab)

chi2, p, _, _ = stats.chi2_contingency(ctab)

print('chi2 :', chi2, ',p :', p)
# chi2 : 2.7669512025956684, p : 0.25070568406521365
# 해석 : p-value : 0.2507 > 0.05 이므로 귀무가설 채택
# 부모의 학력 수준이 자녀의 진학여부와 관련이 없다.

print('---------------')
# 카이제곱 문제2) jikwon_jik과 jikwon_pay 간의 관련성 분석. 가설검정하시오.
#   예제파일 : MariaDB의 jikwon table 
#   jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#   jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
#   조건 : NA가 있는 행은 제외한다.

# 귀무 : 직급과 연봉은 관련이 없다.
# 대립 : 직급과 연봉은 관련이 있다.

import MySQLdb
import numpy as np
import pickle
try:
    with open('mydb.dat', 'rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('err1 :', e)

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = "select jikwon_jik, jikwon_pay from jikwon"
    
    df = pd.read_sql(sql, conn)
    df.columns = '직급', '연봉'
    #print(df)
    # (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
    # (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
    # df['직급'] = np.where(df['직급'] == '이사', 1, df['직급'])
    # df['직급'] = np.where(df['직급'] == '부장', 2, df['직급'])
    # df['직급'] = np.where(df['직급'] == '과장', 3, df['직급'])
    # df['직급'] = np.where(df['직급'] == '대리', 4, df['직급'])
    # df['직급'] = np.where(df['직급'] == '사원', 5, df['직급'])
    
    df['직급'] = df['직급'].apply(
        lambda g:1 if g == '이사' else 2 if g == '부장' else 3 \
                    if g == '과장' else 4 if g == '대리' else 5)

    df['연봉'] = np.where(df['연봉'] < 3000, 1, df['연봉'])
    df['연봉'] = np.where((df['연봉'] >= 3000) & (df['연봉'] < 5000), 2, df['연봉'])
    df['연봉'] = np.where((df['연봉'] >= 5000) & (df['연봉'] < 7000), 3, df['연봉'])
    df['연봉'] = np.where(df['연봉'] >= 7000, 4, df['연봉'])
     
    print(df.head(5))
    print()
    ctab = pd.crosstab(index=df['직급'], columns=df['연봉'], dropna=True)  # NA가 있는 행은 제외
    print(ctab)
    
    chi2, p, _, _ = stats.chi2_contingency(ctab)
    print('chi2 :', chi2, ', p :', p) 
    # chi2 : 37.403 ,p : 0.000192 < 0.05



except Exception as e:
    print('err2 :',e)
finally:
    cursor.close()
    conn.close()
    
# 해석 : p-value : p : 0.00019211 < 0.05 이므로 귀무가설 기각
# 직급과 연봉은 관련이 있다. 는 대립가설 채택
