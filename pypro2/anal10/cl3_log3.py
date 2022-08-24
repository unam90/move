# LogisticRegression 클래스 사용
# Pima 인디언 관련 당뇨병 데이터를 보고 당뇨병을 예측하는 분류 모델
# Pregnancies: 임신 횟수
# Glucose: 포도당 부하 검사 수치
# BloodPressure: 혈압(mm Hg)
# SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
# Insulin: 혈청 인슐린(mu U/ml)
# BMI: 체질량지수(체중(kg)/키(m))^2
# DiabetesPedigreeFunction: 당뇨 내력 가중치 값
# Age: 나이
# Outcome: 클래스 결정 값(0 또는 1)

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib, pickle
from sklearn.metrics import accuracy_score

names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv('pima-indians-diabetes.data.csv', header=None, names=names)
print(df.head(3), df.shape)  # (768, 9)

arr = df.values
print(arr[:3])
x = arr[:, 0:8]
y = arr[:, 8]
print(x.shape) # (768, 8)
print(y.shape) # (768,)
print(set(y))

print()
# train / test split(overfitting 방지 위해)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (537, 8) (231, 8) (537,) (231,)

# 모델 생성
model = LogisticRegression()
model.fit(x_train, y_train)

print('예측값:', model.predict(x_test[:10]))
print('실제값:', y_test[:10])
print('예측값과 실제값이 다른 경우의 갯수:', (model.predict(x_test) !=y_test).sum())
print('분류 정확도:', model.score(x_train, y_train))  # train으로 정확도 확인 / 분류 정확도: 0.78398
pred = model.predict(x_test)
print('분류 정확도:', accuracy_score(y_test, pred))  # test로 정확도 확인 / 분류 정확도: 0.74891

# 학습된 모델 저장
joblib.dump(model, 'pima_model.sav')
# pickle.dump(model, open('pima_model.sav', 'wb'))
del model

# 학습된 모델 읽기
mymodel = joblib.load('pima_model.sav')
# mymodel = pickle.load(open('pima_model.sav', 'rb'))
mypred = mymodel.predict(x_test)
print('분류 정확도:', accuracy_score(y_test, mypred))

# 예측
# print(x_test[:1])
new_data = [[  7., 136., 74., 26., 135., 26., 0.647, 51. ]]
print('새로운 값 예측 결과:', mymodel.predict(new_data))



