# Two-way ANOVA (이원배치 분산분석) : 목적이 되는 요인이 2개 이상인 경우
# 두 요인의 교호작용(한 요인의 효과가 다른 요인의 수준에 의존하는 경우)을 검정할 수 있는 특징이 있다.
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import urllib.request
import statsmodels.api as sm

# 태아수와 관측자수에 따른 태아의 머리둘레 데이터 사용
# 귀무 : 태아수와 관측자수는 태아의 머리둘레 평균과 관련이 없다.
# 대립 : 태아수와 관측자수는 태아의 머리둘레 평균과 관련이 있다.
url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3_2.txt'
data = pd.read_csv(urllib.request.urlopen(url))
print(data.head(3))

# 교호(상호)작용은 빼고 처리함 
ols1 = ols("data['머리둘레'] ~ C(data['태아수']) + C(data['관측자수'])", data=data).fit()  # 범주형일 때 C를 붙여준다.
result = sm.stats.anova_lm(ols1, typ=2)
print(result)

# 교호(상호)작용을 처리함 
ols2 = ols("머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)", data=data).fit()  # 범주형일 때 C를 붙여준다.
result2 = sm.stats.anova_lm(ols2, typ=2)
print(result2)
# 해석 : p-value:3.295509e-01 > 0.05 이므로 귀무가설 채택 
# 태아수와 관측자수는 태아의 머리둘레 평균과 관련이 없다.

