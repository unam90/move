# [two-sample t 검정 : 문제2]  
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 
# 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
# 남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
# 여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4

import random
import numpy as np
import pandas as pd
import scipy.stats as stats

man = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
woman = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

random.seed(123)
man = pd.to_numeric(random.sample(man, 15))  # 비복원 추출
woman = pd.to_numeric(random.sample(woman, 15))  

print(np.mean(man), ' ', np.mean(woman))  # 3.220000000000001   3.273333333333334
print(stats.shapiro(man).pvalue, ' ', stats.shapiro(woman).pvalue)
# 0.0376255065202713   0.0014988253824412823 < 0.05 이므로 정규성 만족 못함

# 정규성 만족인 경우
print(stats.ttest_ind(man, woman))
# Ttest_indResult(statistic=-0.06435846357954361, pvalue=0.949142089962745)
# 해석 : pvalue=0.9491 > 0.05 이므로 귀무가설 채택

# 정규성 불만족인 경우
print(stats.wilcoxon(man, woman))
# WilcoxonResult(statistic=59.0, pvalue=0.97796630859375)

print('----------------------------')
# [대응표본 t 검정 : 문제4]
# 어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 비슷하게 유지되고 있다고 말하고 있다. 
# 이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 
# 점수는 학생 번호 순으로 배열되어 있다.
#    중간 : 80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80
#    기말 : 90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95
# 그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?

# 귀무 : 매년 학기 내 치뤄지는 시험성적의 결과에 따른 학업능력에 변화가 없다.
# 대립 : 매년 학기 내 치뤄지는 시험성적의 결과에 따른 학업능력에 변화가 있다.

mid = [80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80]
fin = [90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95]
print(np.mean(np.array(mid)))  # 74.16
print(np.mean(np.array(fin)))  # 81.66
print(stats.ttest_rel(mid, fin))
# Ttest_relResult(statistic=-2.6281127723493993, pvalue=0.023486192540203194)
# 해석 : pvalue=0.0234 < 0.05 이므로 귀무가설 기각, 대립가설 채택
# 매년 학기 내 치뤄지는 시험성적의 결과에 따른 학업능력에 변화가 있다.






