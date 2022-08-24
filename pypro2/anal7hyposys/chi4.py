# 동질성 검정 - 두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법이다. 
# 두 집단 이상에서 각 범주(집단) 간의 비율이 서로 동일한가를 검정하게 된다. 
# 두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법이다.
# 동질성 검정실습1) 교육방법에 따른 교육생들의 만족도 분석 - 동질성 검정 survey_method.csv

# 귀무 : 교육방법에 따른 교육생들의 만족도에 차이가 없다.
# 대립 : 교육방법에 따른 교육생들의 만족도에 차이가 있다.

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('testdata/survey_method.csv')
print(data.head(3))

print(data['method'].unique())  # [1 2 3]
print(data['survey'].unique())  # [1 2 3 4 5]

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.index=['방법1', '방법2', '방법3']
ctab.columns=['매우만족', '만족', '보통', '불만족', '매우불만족']
print(ctab)

chi2, p, ddof, _ = stats.chi2_contingency(ctab)
msg = '검정통계량 chi2:{}, p-value:{}, df:{}'
print(msg.format(chi2, p, ddof))

# 해석 : p-value:0.58645 > 0.05 이므로 귀무가설 채택
# 교육방법에 따른 교육생들의 만족도에 차이가 없다.

print('---------------------------------------')
# 동질성 검정 실습2) 연령대별 sns 이용률의 동질성 검정
# 20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용 현황을 조사한 자료를 바탕으로 
# 연령대별로 홍보 전략을 세우고자 한다.
# 연령대별로 이용 현황이 서로 동일한지 검정해 보도록 하자

# 귀무(H0) : 연령대별 sns 서비스별 이용률은 동일하다.
# 대립(H1) : 연령대별 sns 서비스별 이용률은 동일하지 않다.

snsdata = pd.read_csv('testdata/snsbyage.csv')
print(snsdata.head(3), len(snsdata))

print(set(snsdata['age']))  # {1, 2, 3}
print(set(snsdata['service']))  # {'T', 'E', 'C', 'K', 'F'}

crotab = pd.crosstab(index=snsdata['age'], columns=snsdata['service'])
print(crotab)

chi2, p, ddof, _ = stats.chi2_contingency(crotab)
print('chi2:{}, p-value:{}, df:{}'.format(chi2, p, ddof))

# 해석 : p-value: 1.1679064204212775e-18 < 0.05 이므로 귀무가설을 기각하고 대립가설을 채택.
# 연령대별 sns 서비스별 이용률은 동일하지 않다.




