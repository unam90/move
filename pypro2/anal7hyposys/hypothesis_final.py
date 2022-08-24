# 가설검정 정리 : chi2, t-test, ANOVA
# jikwon 테이블을 사용
import MySQLdb 
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('connection err : ', e)

conn = MySQLdb.connect(**config)
cursor = conn.cursor()

print('교차분석(이원카이제곱검정): 각 부서(범주형)와 직원평가점수(범주형) 간의 관련성 여부')
# 귀무 : 각 부서와 직원 평가점수 간에 관련이 없다.
# 대립 : 각 부서와 직원 평가점수 간에 관련이 있다. 

df = pd.read_sql('select * from jikwon', conn)
print(df.head(3))
buser = df['buser_num']
rating = df['jikwon_rating']

# 교차표 작성
ctab = pd.crosstab(buser, rating)
print(ctab)

chi, p, df, _ = stats.chi2_contingency(ctab)
print('chi:{}, p:{}, df:{}'.format(chi, p, df))
# 해석 : 유의확률 p-value:0.2906 > 0.05(유의수준)이므로 귀무가설 채택
# 각 부서와 직원 평가점수 간에 관련이 없다.

print()
print('평균차이분석(t-test 독립표본검정): 10, 20번 부서(범주형)의 평균연봉(연속형) 값의 차이가 있는가')
# 귀무 : 두 부서 간 연봉 평균의 차이가 없다.
# 대립 : 두 부서 간 연봉 평균의 차이가 있다.
df_10 = pd.read_sql('select buser_num , jikwon_pay from jikwon where buser_num = 10', conn) 
df_20 = pd.read_sql('select buser_num , jikwon_pay from jikwon where buser_num = 20', conn) 
buser10 = df_10['jikwon_pay']
buser20 = df_20['jikwon_pay']

print('평균 : ', np.mean(buser10), ' ', np.mean(buser20))  # 5414.285   4908.333
t_result = stats.ttest_ind(buser10, buser20)
print(t_result)
# Ttest_indResult(statistic=0.4585177708256519, pvalue=0.6523879191675446)
# 해석 : p-value : 0.6523 > 0.05 이므로 귀무가설 채택
# 두 부서 간 연봉 평균의 차이가 없다. 

print()
print('분산분석(ANOVA): 각 부서(범주형, 요인 1개에 4그룹이 존재)의 평균연봉(연속형) 값의 차이가 있는가')
# 귀무 : 4개의 부서 간 연봉 평균의 차이가 없다.
# 대립 : 4개의 부서 간 연봉 평균의 차이가 있다.

df3 = pd.read_sql('select buser_num, jikwon_pay from jikwon', conn)
buser = df3['buser_num']
pay = df3['jikwon_pay']

gr1 = df3[df3['buser_num']==10]['jikwon_pay']
gr2 = df3[df3['buser_num']==20]['jikwon_pay']
gr3 = df3[df3['buser_num']==30]['jikwon_pay']
gr4 = df3[df3['buser_num']==40]['jikwon_pay']

plt.boxplot([gr1, gr2, gr3, gr4])
plt.show()

# 방법1
f_sta, pv = stats.f_oneway(gr1, gr2, gr3, gr4)
print(f_sta, pv)  # p-value : 0.74544 > 0.05 이므로 귀무가설 채택
# 4개의 부서 간 연봉 평균의 차이가 없다.

# 방법2
lm = ols('jikwon_pay ~ C(buser_num)', data=df3).fit()  # ols(예측하고자 하는 칼럼 이름 ~ 원인이되는 칼럼 이름(+로 연결))
result = anova_lm(lm, typ=2)
print(result)  # p-value : 0.7454 

print()
print('사후 검정')
from statsmodels.stats.multicomp import pairwise_tukeyhsd
pt = pairwise_tukeyhsd(df3.jikwon_pay, df3.buser_num)
print(pt)
pt.plot_simultaneous()
plt.show()



















