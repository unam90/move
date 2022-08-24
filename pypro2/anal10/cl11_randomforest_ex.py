# [Randomforest 문제2]
# 중환자 치료실에 입원 치료 받은 환자 200명의 생사 여부에 관련된 자료다.
# 종속변수 STA(환자 생사 여부)에 영향을 주는 주요 변수들을 이용해 검정 후에 해석하시오. 
# 모델 생성 후 입력자료 및 출력결과는 Django를 사용하시오.
# 예제 파일 : https://github.com/pykwon  ==>  patient.csv
#
# <변수설명>
#   STA : 환자 생사 여부 (0:생존, 1:사망)
#   AGE : 나이
#   SEX : 성별
#   RACE : 인종
#   SER : 중환자 치료실에서 받은 치료
#   CAN : 암 존재 여부
#   INF : 중환자 치료실에서의 감염 여부
#   CPR : 중환자 치료실 도착 전 CPR여부
#   HRA : 중환자 치료실에서의 심박수
#
# 참고 : 중요 변수 알아보기
# print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))
#
# def plot_feature_importances(model):   # 특성 중요도 시각화
#     n_features = x.shape[1]
#     plt.barh(range(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), x.columns)
#     plt.xlabel("attr importances")
#     plt.ylabel("attr")
#     plt.ylim(-1, n_features)
#     plt.show()
#     plt.close()
#
# plot_feature_importances(model)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 표준화 지원 클래스
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('patient.csv')
print(df.head(3))
print(df.isnull().sum())  # 0

df_x = df[['AGE', 'SEX', 'RACE','SER', 'CAN','INF','CPR','HRA']]
print(df_x[:3], df_x.shape)  # (200, 8)

df_y = df['STA']
print(df_y[:3])

print()
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, random_state =12) 
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)  # (150, 8) (50, 8) (150,) (50,)

# model
model = RandomForestClassifier(criterion='entropy', n_estimators=100)
model = model.fit(train_x, train_y)

# pred 
pred = model.predict(test_x)
print('예측값:', pred[:10])
print('실제값:', np.array(test_y[:10]))

# 분류 정확도
print('acc:', sum(test_y ==pred) / len(test_y))
print('acc:', accuracy_score(test_y, pred))

print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))

def plot_feature_importances(model):   # 특성 중요도 시각화
    n_features = df_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df_x.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()

plot_feature_importances(model)

import joblib

joblib.dump(model, './patientmodel')
