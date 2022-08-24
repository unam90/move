# 날씨 정보로 다음날 비가 올 지 예측하는 분류 모델
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics._scorer import accuracy_score

data = pd.read_csv('weather.csv')
print(data.head(2), data.shape)  # (366, 12)

data2 = pd.DataFrame()
data2 = data.drop(['Date', 'RainToday'], axis=1)
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})
print(data2.head(2), data2.shape)  # (366, 10)
print(data2.RainTomorrow.unique())  # [1 0]

print()
# 분류모델의 overfitting(과적합) 방지를 목적으로 train(학습용) / test(검정용) dataset을 작성
train, test = train_test_split(data2, test_size=0.3, random_state=42)
print(train.shape, test.shape)  # (256, 10) (110, 10)

# 분류 모델 
# my_formula = 'RainTomorrow ~ MinTemp+MaxTemp+Rainfall...'
my_formula = 'RainTomorrow ~ ' + '+'.join(train.columns.difference(['RainTomorrow']))
print(my_formula)
model = smf.glm(formula = my_formula, data=train, family=sm.families.Binomial()).fit()

print(model.summary())

print('예측값:', np.rint(model.predict(test)[:10].values))  # np.around = np.rint
print('실제값:', test['RainTomorrow'][:10].values)

# 분류 정확도 
# conf_mat = model.pred_table()  # AttributeError: 'GLMResults' object has no attribute 'pred_table' 
# glm()은 pred_table을 지원하지 않음
# print('confusion matrix: \n', conf_mat)

pred = model.predict(test)
print('정확도:', accuracy_score(test['RainTomorrow'], np.around(pred)))  # 정확도: 0.87272

print()
model2 = smf.logit(formula=my_formula, data=train).fit()
conf_mat = model2.pred_table()
print('confusion matrix: \n', conf_mat)
# confusion matrix: 
# [[197.   9.]
# [ 21.  26.]]

print('정확도:', (conf_mat[0][0] + conf_mat[1][1]) / len(train))  # 정확도: 0.87109










 











