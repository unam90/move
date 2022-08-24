from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# [로지스틱 분류분석 문제3] 
# Kaggle.com의 https://www.kaggle.com/truesight/advertising.csv file을 사용
#   참여 칼럼 : 
#   Daily Time Spent on Site : 사이트 이용 시간 (분)
#   Age : 나이,
#   Area Income : 지역 소득
#   Daily Internet Usage:일별 인터넷 사용량(분),
#   Clicked Ad : 광고 클릭 여부 (0, 1) 
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# ROC 커브와 AUC 출력

data = pd.read_csv('advertising.csv', usecols=[0,1,2,3,9])
print(data.head(3), len(data))
print(data.columns)

# 칼럼명 변경
data.columns =['사이트이용시간', '나이', '지역소득', '일별인터넷사용량', '광고 클릭여부']
print(data.head(3))

x = data.iloc[:, :4]
y = data.iloc[:, [4]]
print(x[:4])
print(y[:4])

# 모델 생성
model = LogisticRegression().fit(x, y)
y_hat = model.predict(x)
print('분류결과:', y_hat[:10])
print('실제값:', y[:10])
print()
print(confusion_matrix(y, y_hat))
acc = (464 + 433) / (464+36+67+433)  # acc: 0.897
precision = 464 / (464+67)  # precision: 0.8738229755178908
recall = 464 / (464+36)  # recall: 0.928
fallout = 67 / (67+36)  # fallout: 0.6504854368932039
print('acc:', acc)
print('precision:', precision)
print('recall:', recall)
print('fallout:', fallout)

print()
from sklearn import metrics
ac_sco = metrics.accuracy_score(y, y_hat)
print('ac_sco:', ac_sco) 
print()
cl_rep = metrics.classification_report(y, y_hat)
print('cl_rep:', cl_rep)

print()
# 판별함수(결정 함수, 불확실성 추정함수. ROC 커브를 그릴 때 판별 경계선으로 사용)
f_value = model.decision_function(x)
# print('f_value:', f_value)
print()
fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
# print('fpr:', fpr)
# print('tpr:', tpr)
# print('분류임계값:', thresholds)

# 시각화 
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0,1],[0,1],'k--', label='classifier line(AUC:0.5)')  
plt.plot([fallout],[recall], 'ro', ms=10)  # 위양성율, 재현율값 출력
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

# AUC
print('AUC', metrics.auc(fpr, tpr))
# AUC 0.9580599999999999


