# 산탄데르 고객 만족(Santander Customer Satisfaction)
# 캐글의 산탄데르 고객 만족 데이터 세트에 대해 고객 만족 여부를 XGBoost 와 LightGBM 으로 예측해보자. 
# 이 데이터 세트는 370개의 feature로 주어진 데이터 세트 기반으로 고객 만족 여부를 예측하는 것인데, 
# feature 이름이 모두 익명으로 처리돼 이름만으로는 어떤 속성인지는 추정할 수 없다. 
# TARGET 레이블이 1이면 불만, 0이면 만족한 고객이다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cust_df = pd.read_csv('train.csv', encoding='latin-1')
print(cust_df.head(3), cust_df.shape)  # (76020, 371)
print(cust_df.info())  # 111개의 float63형과 260개의 int64형으로 모든 피처가 숫자형이며, Null 값은 없다.
print(cust_df['TARGET'].value_counts())  # 은행에 대한 고객의 만족/불만족 비율 
# 0(만족):73012개, 1(불만족):3008
print()
unstisfied_cnt = cust_df[cust_df['TARGET']==1].TARGET.count()
total_cnt = cust_df.TARGET.count()
print(f'불만족 비율: {np.round((unstisfied_cnt/total_cnt*100), 2)}%')  # 불만족 비율: 3.96%

print()
print(cust_df.describe())

# var3열에 이상치를 2로 대체
cust_df['var3'].replace(-999999, 2, inplace=True)
# ID열 제거
cust_df.drop('ID', axis=1, inplace=True) 

# feature 세트와 label 세트 분리. 레이블 칼럼은 DataFrame의 맨 마지막에 위치해 칼럼 위치 -1로 분리
X_features = cust_df.iloc[:, :-1]  # 마지막열 전까지가 feature
y_labels = cust_df.iloc[:, -1]  # 마지막열이 label
print(f'피처 데이터 shape: {X_features.shape}') # (76020, 369)
print('피처 데이터 shape: {0}'.format(X_features.shape))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# (60816, 369) (15204, 369) (60816,) (15204,)

train_cnt = y_train.count()
test_cnt = y_test.count()
print(f'학습 세트 shape: {X_train.shape}, 테스트 세트 shape: {X_test.shape}')
print('학습 세트 레이블 값 분포 비율')
print(y_train.value_counts()/train_cnt)
print('\n테스트 데이터 세트 레이블 값 분포 비율')
print(y_test.value_counts()/test_cnt)
# 학습 데이터와 테스트 데이터 세트 모두 불만족 비율이 4% 정도로 만들어졌다.

# model
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_clf = XGBClassifier(n_estimator=500, random_state=12)
xgb_clf.fit(X_train, y_train, 
            early_stopping_rounds=3,  # 3회 돌고, 조기 중단(모델 학습 중단 parameter) 
            eval_metric='auc',        # accuracy값이 증가하다가 3번까지 학습했는데 안늘어나면 중단 
            eval_set=[(X_train, y_train), (X_test, y_test)])  # 학습하면서 평가도 하는 parameter
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1], average='macro')
print(f'ROC AUC: {np.round(xgb_roc_score, 4)}')  # ROC AUC: 0.8413

print()
pred = xgb_clf.predict(X_test)
print('예측값:', pred[:10])
print('실제값:', y_test[:10].values)
from sklearn import metrics
acc = metrics.accuracy_score(y_test, pred)
print('분류정확도:', acc)  # 분류정확도: 0.958103



