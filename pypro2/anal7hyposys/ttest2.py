# 두 집단의 가설검정 – 실습 시 분산을 알지 못하는 것으로 한정하겠다.
# * 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.
# 실습) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정
male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]

# 귀무 : 남녀 두 집단 간 파이썬 시험의 평균에 차이가 없다.
# 대립 : 남녀 두 집단 간 파이썬 시험의 평균에 차이가 있다.


from scipy import stats
import pandas as pd
from numpy import mean, average

print(mean(male))   # 83.8
print(mean(female)) # 72.24

# 독립표본 t검정 수행
two_sample = stats.ttest_ind(male, female)
print(two_sample)
# Ttest_indResult(statistic=1.233193127514512, pvalue=0.2525076844853278)
# 해석 : pvalue=0.2525 > 0.05 이므로 귀무가설 채택, 대립가설 기각
# 남녀 두 집단 간 파이썬 시험의 평균에 차이가 없다.

print()
print('실습 2')
# 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv
# 귀무 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.
# 대립 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 있다.

data = pd.read_csv('testdata/two_sample.csv')
print(data.head(3))
ms = data[['method', 'score']]  # method와 score 칼럼만 가져오기 
print(ms.head(3))

m1 = ms[ms['method'] == 1]  # method1 => m1
m2 = ms[ms['method'] == 2]  # method2 => m2
# print(m1)
score1 = m1['score']  
score2 = m2['score']
print(score1.isnull().sum())  # score1의 null값 갯수 확인
print(score2.isnull().sum())

# sco1 = score1.fillna(0)  # NaN : dropna() or 0으로 채우기
sco1 = score1.fillna(score1.mean())  # score1의 null값을 평균값으로 채우기
sco2 = score2.fillna(score2.mean())

print('정규성 확인')
# 시각화로 확인
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(sco1, kde=True, color='r')
sns.histplot(sco2, kde=True)
plt.show()

print(stats.shapiro(sco1).pvalue)  # 0.36799 > 0.05 정규성 만족
print(stats.shapiro(sco2).pvalue)  # 0.67141 > 0.05 정규성 만족

print('등분산성 확인')
print(stats.levene(sco1, sco2).pvalue)    # 모수 검정인 경우   0.4568 > 0.05 등분산성 만족
print(stats.fligner(sco1, sco2).pvalue)
print(stats.bartlett(sco1, sco2).pvalue)  # 비모수 검정인 경우

result = stats.ttest_ind(sco1, sco2, equal_var=True)  # 등분산성 만족인 경우  
# result = stats.ttest.ind(sco1, sco2, equal_var=False)  # 등분산성 만족이 아닌 경우 
print(result)
# Ttest_indResult(statistic=-0.19649386929539883, pvalue=0.8450532207209545)
# 해석 : pvalue=0.8450 > 0.05 이므로 귀무가설 채택
# 두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.

# 참고 : 정규성을 만족하지 않은 경우 
# print(stats.wilcoxon(sco1, sco2))  # 크기가 같은 경우
print(stats.mannwhitneyu(sco1, sco2))  # 크기가 다른 경우











