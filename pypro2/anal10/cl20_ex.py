# [SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습
# https://www.kaggle.com/zhaoyingzhu/heartcsv
# https://github.com/pykwon/python/tree/master/testdata_utf8   Heartcsv
# Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
# 각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
# dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
# 임의의 값을 넣어 분류 결과를 확인하시오.     
# 정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ
#
# feature 칼럼 : 문자 데이터 칼럼은 제외
# label 칼럼 : AHD(중증 심장질환)
#
# 데이터 예)
# "","Age","Sex","ChestPain","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca","Thal","AHD"
# "1",63,1,"typical",145,233,1,2,150,0,2.3,3,0,"fixed","No"
# "2",67,1,"asymptomatic",160,286,0,2,108,1,1.5,2,3,"normal","Yes"
# ...

# SVM으로 분류모델 만들기
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Heart.csv')
print(data.head(3), data.shape)
print(data.info())
print(data.isnull().sum())

data = data.drop(['ChestPain', 'Thal'], axis=1)  # object type 제외
data = data.fillna(data.mean())
print(data.isnull().sum())
print(data.head(3))

feature = data.loc[:, data.columns!='AHD']
# feature = data.drop(['AHD'], axis=1)
print(feature[:5])
label = data['AHD']
print(label[:5])

# label을 dummy화 
label = label.map({'No':0, 'Yes':1})
print(label[:3])

# train, test로 나누기
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3, random_state=1)
print(feature_train.shape, feature_test.shape, label_train.shape, label_test.shape)
# (212, 12) (91, 12) (212,) (91,)

print()
# model
model = svm.SVC(C=1).fit(feature_train, label_train)

pred = model.predict(feature_test)
print('예측값:', pred[:10])
print('실제값:', label_test[:10].values)

# accuracy
acc = metrics.accuracy_score(label_test, pred)
print('정확도:', acc)

# 새로운 예측
new_test = feature_test[:2].copy()
print(new_test)
new_test['Age'] = 10
new_test['Sex'] = 0
print(new_test)

new_pred=model.predict(new_test)
print(new_pred)