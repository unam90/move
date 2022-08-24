# 회귀분석 문제 3) 
# kaggle.com에서 carseats.csv 파일을 다운 받아 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.

import statsmodels.formula.api as smf
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("testdata/Carseats.csv")
print(df.head(3))
print(df.info())

df = df.drop([df.columns[6], df.columns[9], df.columns[10]], axis=1)  # 범주형은 제외
print(df.columns)

print('상관관계 확인')
print(df.corr())

print()
model = smf.ols(formula = 'Sales ~ Income + Advertising + Price + Age', data = df).fit()
print(model.summary())

# 모델 저장 후 읽기
import joblib
joblib.dump(model, 'aaa.model')
del model
model = joblib.load('aaa.model')

print()
print('# 회귀분석모형의 적절성을 위한 선행 조건도 체크 ---')
fitted = model.predict(df.iloc[:, [0,2,3,5,6]])
residual = df['Sales'] - fitted  # 잔차
print('residual : ', residual)

print('선형성 ---')
sns.regplot(fitted, residual, lowess = True, line_kws = {'color':'red'})
plt.plot([fitted.min(), fitted.max()], [ 0, 0], '--', color='blue')
plt.show()  # 미흡하지만 선형성을 만족

print('정규성 ---')
sr = scipy.stats.zscore(residual)
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x, y)
plt.plot([-3, 3],[-3, 3], '--', color='green')
plt.show()

print('잔차의 정규성 : ', scipy.stats.shapiro(residual))
# ShapiroResult(statistic=0.994922399520874, pvalue=0.2127407342195511)
# pvalue=0.2127407342195511 > 0.05 # 정규성을 만족

print('독립성 ---')
# Durbin-Watson : 1.931  - 0 ~ 4 사이의 값을 갖는데 2에 가까우면 자기상관이 없다. 독립적이다.

print('등분산성 ---')
sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess = True, line_kws={'color':'red'})
plt.show()  # 등분산성 만족

print('다중 공선성 ---')
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(df.iloc[:, [0,2,3,5,6]].values, i) 
                      for i in range(df.iloc[:, [0,2,3,5,6]].shape[1])]
print(vifdf)

print('새로운 값으로 예측')
new_df = pd.DataFrame({'Income':[35, 62],'Advertising':[6, 3],
                       'Price':[100, 60],'Age':[33, 40]})
new_pred = model.predict(new_df)
print('예측 결과 : ', new_pred.values)
