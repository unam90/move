# SVM으로 논리 연산 (저차원을 고차원으로 만들어서 해결)

x_data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0],
]

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

feature = []
label = []

for row in x_data:
    p = row[0]
    q = row[1]
    r = row[2]
    feature.append([p,q])
    label.append(r)
    
print(feature)
print(label)

# model
# model = LogisticRegression()  # 선형분류 모델 (정확도 0.5밖에 안나옴)
model = svm.SVC()  # svm 정확도 1.0 

model.fit(feature, label)

pred = model.predict(feature)
print('예측값:', pred)
print('실제값:', label)

# 정확도 
acc = metrics.accuracy_score(label, pred)
print('acc:', acc)
print('report: \n', metrics.classification_report(label, pred))
