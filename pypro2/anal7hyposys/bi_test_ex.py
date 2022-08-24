# [이항분포 검정 : 문제]
# seaborn이 제공하는 tips datasets으로 어느 식당의 판매 기록자료를 구할 수 있다. 
# 하나의 행이 한 명의 손님을 나타낸다고 가정하자. 
# 열에는 성별(sex), 흡연유무(smoker), 점심/저녁(time) 등을 나타내는 데이터가 있다. 
# 이항 검정을 사용하여 아래의 문제를 해결하시오.
# 여자 손님 중 비흡연자가 흡연자보다 60% 이상 많다고 할 수 있는가?
# 저녁에 오는 여자 손님 중 비흡연자가 흡연자보다 80% 이상 많다고 할 수 있는가?

import pandas as pd
import scipy.stats as stats
import seaborn as sns
tips = sns.load_dataset("tips")
print(tips.head(3))
print()
print('여자 손님 중 비흡연자가 흡연자보다 60% 이상 많다고 할 수 있는가?')
# 귀무 : 여자 손님 중 비흡연자가 흡연자보다 60% 이상 많다고 할 수 없다.
# 대립 : 여자 손님 중 비흡연자가 흡연자보다 60% 이상 많다고 할 수 있다.
f_data = tips[tips.sex == 'Female']
print(f_data.head(3), len(f_data))
print()
ctab = pd.crosstab(index=f_data['smoker'], columns='count')
print(ctab)
print()
result1 = stats.binom_test([54, 33], p=0.6, alternative='greater')
print('p-value:', result1)  # p-value: 0.39070 > 0.05 이므로 귀무가설 채택
# 여자 손님 중 비흡연자가 흡연자보다 60% 이상 많다고 할 수 없다.

print()
print('저녁에 오는 여자 손님 중 비흡연자가 흡연자보다 80% 이상 많다고 할 수 있는가?')
# 귀무 : 저녁에 오는 여자 손님 중 비흡연자가 흡연자보다 80% 이상 많다고 할 수 없다.
# 대립 : 저녁에 오는 여자 손님 중 비흡연자가 흡연자보다 80% 이상 많다고 할 수 있다.
f_data2 = f_data[f_data.time == 'Dinner']
print(f_data2.head(3), len(f_data2))

ctab2 = pd.crosstab(index=f_data2['smoker'], columns='count')
print(ctab2)

print()
result2 = stats.binom_test([29, 23], p=0.8, alternative='greater')
print('p-value:', result2)  # p-value: 0.9999 > 0.05 이므로 귀무가설 채택
# 저녁에 오는 여자 손님 중 비흡연자가 흡연자보다 80% 이상 많다고 할 수 없다.
