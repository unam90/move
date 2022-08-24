# 이원카이제곱 - 교차분할표 이용
# : 두 개 이상의 변인(집단 또는 범주)을 대상으로 검정을 수행한다.
# 분석대상의 집단 수에 의해서 독립성 검정과 동질성 검정으로 나뉜다.

# 독립성(관련성) 검정
# - 동일 집단의 두 변인(학력수준과 대학진학 여부)을 대상으로 관련성이 있는가 없는가?
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다.
# 실습 : 교육수준과 흡연율 간의 관련성 분석 : smoke.csv

# 귀무가설 : 교육수준과 흡연율 간에 관련이 없다.(독립이다.)
# 대립가설 : 교육수준과 흡연율 간에 관련이 있다.(독립이 아니다.)

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/smoke.csv')
print(data.head(3))
print(data['education'].unique())  # [1 2 3]
print(data['smoking'].unique()) # [1 2 3]

ctab = pd.crosstab(index=data['education'], columns=data['smoking'])  # , normalize=True는 비율을 보여줌
ctab.index = ['고졸','대졸','대학원졸']
ctab.columns = ['과흡연', '보통', '노담']
print(ctab)

chi_result = [ctab.loc['고졸'], ctab.loc['대졸'], ctab.loc['대학원졸']]
print(chi_result)

# chi2, p, ddof, exp = stats.chi2_contingency(chi_result)

chi2, p, ddof, exp = stats.chi2_contingency(ctab)

# print(exp) 카이제곱검정 예측값이지만 필요하지 않다.
print('chi2:{}, p-value:{}, df:{}'.format(chi2, p, ddof))
# chi2:18.910915739853955, p-value:0.0008182572832162924, df:4 = (3-1)*(3-1)

# 해석 : p-value:0.000818 < 0.05 이므로 귀무가설을 기각하고 대립가설을 채택
# 교육수준과 흡열율 간에 관련이 있다. (독립이 아니다.)

print('-------------------------------------------')
# 실습) 국가전체와 지역에 대한 인종 간 인원수로 독립성 검정 실습
# 두 집단(국가전체 - national, 특정지역 - la)의 인종 간 인원수의 분포가 관련이 있는가?
# 귀무 : 국가전체와 지역에 대한 인종 간 인원수는 관련이 없다. (독립적이다.)
# 대립 : 국가전체와 지역에 대한 인종 간 인원수는 관련이 있다. (독립적이지 않다.)

national = pd.DataFrame(["white"] * 100000 + ["hispanic"] * 60000 +
                        ["black"] * 50000 + ["asian"] * 15000 + ["other"] * 35000)
la = pd.DataFrame(["white"] * 600 + ["hispanic"] * 300 + ["black"] * 250 +
                  ["asian"] * 75 + ["other"] * 150)

print(national.head(3))
print(la.head(3))
na_table = pd.crosstab(index=national[0], columns='count')
la_table = pd.crosstab(index=la[0], columns='count')
na_table['count_la'] = la_table['count']
print(na_table)

chi2, p, ddof, exp = stats.chi2_contingency(na_table)
print(chi2, p)
# 18.099524243141698 0.0011800326671747886
# 해석 : p-value 0.00118 < 0.05 이므로 귀무가설 기각 
# 국가전체와 지역에 대한 인종 간 인원수는 관련이 있다. 라는 대립가설 채택.

     