import numpy as np
import scipy.stats as stats
import pandas as pd

# [one-sample t 검정 : 문제1]  
# 영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다.  
# 한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다.  
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간을 수집하여 다음의 자료를 얻었다.  
# 한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
#    305 280 296 313 287 240 259 266 318 280 325 295 315 278

# 귀무가설 : 신형 백열전구의 수명의 평균은 300시간이다.
# 대립가설 : 신형 백열전구의 수명의 평균은 300시간이 아니다.
data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
print(np.mean(data)) # 표본의 평균:289.7857142857143
res = stats.ttest_1samp(data, popmean=300)
print(res)
# Ttest_1sampResult(statistic=-1.556435658177089, pvalue=0.143606254517609)
# pvalue=0.143 > 0.05보다 크므로 귀무가설 채택
# 신형 백열전구의 수명의 평균이 300시간이라고 주장할 통계적 근거가 존재한다.


print()
# [one-sample t 검정 : 문제2]
# 실습 예제) 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. 
# A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다. 
# 실습 파일 : one_sample.csv

#참고 : time에 공백을 제거할 땐 ***.time.replace("     ", "")

#귀무가설 : A회사에서 생산된 노트북 평균시간은 5.2 시간이다.
#대립가설 : A회사에서 생산된 노트북 평균시간은 5.2 시간이 아니다.
data = pd.read_csv('testdata/one_sample.csv')
print(data.head(7), len(data))
redata = data.time.replace("     ", "").replace("", np.nan).dropna() #109개만 남음
print(np.mean(pd.to_numeric(redata))) #평균: 5.5568807339449515
res2 = stats.ttest_1samp(pd.to_numeric(redata), popmean=5.2)
print(res2)
#statistic=3.9460595666462432, pvalue=0.00014166691390197087
# pvalue=0.00014 < 0.05이므로 귀무가설 기각
# A회사에서 생산된 노트북 평균시간은 5.2 시간이 아니다.


print()
# [one-sample t 검정 : 문제3]
# http://www.price.go.kr에서 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료를 파일로 받아 미용요금을 얻도록 하자.  
# 정부에서는 전국평균 미용요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.

#귀무가설 : 전국평균 미용요금이 15000원이다.
#대립가설 : 전국평균 미용요금이 15000원이 아니다.

data = pd.read_excel('개인서비스지역별_동향_6월.xls', sheet_name="개인서비스지역동향2022-06").T.dropna().iloc[2:,] # T:transpose 
data.columns = ["미용"]
print(data.head(3))
print()
print(np.mean(data.미용))
print(stats.ttest_1samp(data.미용, popmean=15000))

# statistic=4.396456857242187, pvalue=0.000520588880650458
# pvalue=0.00052 < 0.05 이므로 전국평균 미용요금이 15000원이라고 주장할 통계적 근거가 없다.