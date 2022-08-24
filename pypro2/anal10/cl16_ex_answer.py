# [XGBoost 문제] 
# kaggle.com이 제공하는 'glass datasets'
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7가지의 label(Type)로 분리된다.
# RI    Na    Mg    Al    Si    K    Ca    Ba    Fe    Type 
# glass.csv 파일을 읽어 분류 작업을 수행하시오.

import pandas as pd
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/glass.csv")
print(data.columns)
print(data.info())

x = data.drop('Type', axis=1)  # Type 열은 독립변수에서 제외
y = data['Type']
print(set(y))  # {1, 2, 3, 5, 6, 7}

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y[:3], set(y)) # {0, 1, 2, 3, 4, 5}

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                        test_size=0.3, random_state=12)

model = xgb.XGBClassifier(booster='gbtree', max_depth=4, n_estimators=500)
model.fit(x_train,y_train)

print()  
y_pred = model.predict(x_test)
print('실제값 :', y_pred[:5])
print('예측값:', np.array(y_test[:5]))
print('정확도 :', metrics.accuracy_score(y_test, y_pred))

# 중요 변수 시각화
from xgboost import plot_importance
plot_importance(model)
plt.show()