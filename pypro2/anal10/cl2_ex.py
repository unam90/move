# [로지스틱 분류분석 문제1]
# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics._scorer import accuracy_score

data = pd.read_csv('eat.csv')
data = data.loc[(data['요일']=='토') | (data['요일']=='일')]
print(data.head(3), data.shape)  # (21, 3)

# 분류 모델
model = smf.glm(formula = '외식유무 ~ 소득수준', data=data, family=sm.families.Binomial()).fit()
print(model.summary())
print()
# 분류 정확도
pred = model.predict(data)
print('정확도:', accuracy_score(data['외식유무'], np.around(pred)))  # 정확도: 0.90476

print('예측값:', np.around(model.predict(data)[:10].values))
print('실제값:',data['외식유무'][:10].values)

print()
model2 = smf.logit(formula='외식유무 ~ 소득수준', data=data).fit()
matrix = model2.pred_table()
print('confusion matrix: \n', matrix)
# confusion matrix: 
 # [[10.  1.]
 # [ 1.  9.]]

print('정확도:', (matrix[0][0] + matrix[1][1]) / len(data))  # 정확도: 0.904761

print()
print('새로운 값으로 결과 보기')
newdf = data.iloc[:2].copy()
print(newdf)
newdf['소득수준'] = [80, 30]
print(newdf)
print()
new_pred = model2.predict(newdf)
print('new_pred:', np.around(new_pred.values))



