# [XGBoost 문제] 
# kaggle.com이 제공하는 'glass datasets'
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7 가지의 label(Type)로 분리된다.
# RI    Na    Mg    Al    Si    K    Ca    Ba    Fe    
#  Type
#                           ...
# glass.csv 파일을 읽어 분류 작업을 수행하시오.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv('glass.csv')
print(df.head(3), df.shape)  # (214, 10)

# feature, label 분리 
x_feature = df.iloc[:, :-1]
y_label = df.iloc[:, -1]
print('feature shape:{0}'.format(x_feature.shape))  # (214, 9)

from sklearn.preprocessing import LabelEncoder
y_label = LabelEncoder().fit_transform(y_label)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_feature, y_label, test_size=0.3, random_state=777)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (149, 9) (65, 9) (149,) (65,)

# model
import xgboost as xgb
model = xgb.XGBClassifier(booster='gbtree', n_estimators=500, max_depth=4).fit(x_train, y_train)

# 분류 예측
pred = model.predict(x_test)  # test dataset으로 모델 검정
print('예측값:', pred[:10])
print('실제값:', np.array(y_test[:10]))

from sklearn import metrics
acc = metrics.accuracy_score(y_test, pred)
print('분류정확도:', acc)

# 중요 변수 시각화
from xgboost import plot_importance
plot_importance(model)
plt.show()