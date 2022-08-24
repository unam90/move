# b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.
#      - pivot_table을 사용하여 성별 연봉의 평균을 출력
#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))

import MySQLdb
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family = 'malgun gothic')  
plt.rcParams['axes.unicode_minus'] = False

try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('읽기 오류 : ', e)
    
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = """
        select buser_name,jikwon_gen,jikwon_pay
        from jikwon inner join buser 
        on buser_num=buser_no
    """
    cursor.execute(sql)
    
    # 1) DataFrame으로 출력
    df = pd.DataFrame(cursor.fetchall(),
                       columns=['부서','성별','연봉'])
    print(df.head(3))
        
    print('pandas의 sql 처리 기능을 사용해서 읽기')
    df1 = pd.read_sql(sql, conn)
    df1.columns = ['부서','성별','연봉']
    print(df1.head(3))
    
    print(df1.pivot_table(values=['연봉'], index=['성별'], aggfunc=np.mean))    
    
    # 시각화 : 성별 연봉 평균   
    m = df1[df1['성별'] == '남']
    m_pay_mean = m.loc[:,'연봉'].mean()
    f = df1[df1['성별'] == '여']
    f_pay_mean = f.loc[:,'연봉'].mean()
    gen_mpay = [m_pay_mean, f_pay_mean]
    
    plt.bar(range(len(gen_mpay)),gen_mpay, color=['blue','red'])
    plt.title('성별에 따른 연봉 평균')
    plt.xlabel('성별')
    plt.ylabel('연봉평균')
    plt.xticks(range(len(gen_mpay)), labels=['남성','여성'])
    plt.show()
    
    # crosstab 작성
    ctab = pd.crosstab(df1['부서'], df1['성별'])
    print(ctab)
    
except Exception as e:
    print('처리 오류 : ', e)