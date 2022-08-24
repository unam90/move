# [Randomforest 문제3] 
# https://www.kaggle.com/c/bike-sharing-demand/data 에서  
# train.csv를 다운받아 bike_dataset.csv 으로 파일명을 변경한다.
# 이 데이터는 2011년 1월 ~ 2012년 12월 까지 날짜/시간. 기온, 습도, 풍속 등의 정보를 바탕으로 
# 1시간 간격의 자전거 대여횟수가 기록되어 있다. train / test로 분류 한 후 
# 대여횟수에 중요도가 높은 칼럼을 판단하여 feature를 선택한 후, 대여횟수에 대한 회귀예측을 하시오. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree._classes import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics._classification import accuracy_score

df = pd.read_csv('bike_dataset.csv')
print(df.head(3), df.shape)
print(df.info())
print(df.corr()['count'])  # count를 기준으로 상관계수 확인
df.drop(columns=['datetime','season','holiday','registered'], inplace=True)
print(df.head(2), df.shape)  # (10886, 9)


# print('temp:', df['temp'].value_counts())
# print('atemp:', df['atemp'].value_counts())
# print('humidity:', df['humidity'].value_counts())
# print(df.head(3))

print()
from sklearn.model_selection import train_test_split
feature_df = df.drop(['count'], axis='columns')
label_df = df['count']
print(feature_df.head(3))
print(label_df.head(3))

x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 종속변수로 사용할 count 열과 상관관계가 높은 temp, humidity열로 시각화
cols = ['count', 'temp', 'humidity']
sns.pairplot(df[cols])
plt.show()

x = df[['temp', 'humidity']].values
y = df['count'].values
print(x[:3])
print(y[:3])

print()
# RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000, criterion='squared_error').fit(x, y)
print('예측값:', model.predict(x)[:5])
print('실제값:', y[:5])
print('결정계수:', r2_score(y, model.predict(x)))
print()
new_data = np.array([[13, 50]])
print('예측결과:', model.predict(new_data))