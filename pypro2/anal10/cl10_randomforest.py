# Random forest는 ensemble(앙상블) machine learning 모델입니다. 
# 여러개의 decision tree를 형성하고 새로운 데이터 포인트를 각 트리에 동시에 통과시키며, 
# 각 트리가 분류한 결과에서 투표를 실시하여 가장 많이 득표한 결과를 최종 분류 결과로 선택합니다.
# 랜덤 포레스트는 제일 먼저 bagging이라는 과정을 거칩니다.

# 타이타닉 데이터를 사용
# Survived : 0 = 사망, 1 = 생존
# Pclass : 1 = 1등석, 2 = 2등석, 3 = 3등석
# Sex : male = 남성, female = 여성
# Age : 나이
# SibSp : 타이타닉 호에 동승한 자매 / 배우자의 수
# Parch : 타이타닉 호에 동승한 부모 / 자식의 수
# Ticket : 티켓 번호
# Fare : 승객 요금
# Cabin : 방 호수
# Embarked : 탑승지, C = 셰르부르, Q = 퀸즈타운, S = 사우샘프턴

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

df = pd.read_csv('titanic_data.csv')
print(df.head(3))
print(df.columns)
print(df.shape)  # (891, 12)
print(df.isnull().sum())
df = df.dropna(subset=['Pclass','Age','Sex'])
df_x = df[['Pclass','Age','Sex']]
print(df_x[:3], df_x.shape)  # (714, 3)

# Sex(male:1, female:0) : 더미(dummy)변수
from sklearn.preprocessing import LabelEncoder
# df_x.loc[:, 'Sex'] = LabelEncoder().fit_transform(df_x['Sex'])
df_x['Sex'] = df_x['Sex'].apply(lambda x:1 if x=='male' else 0)
print(df_x.head(3))

df_y = df['Survived']  # label
print(df_y[:3])

print()
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, random_state =12) 
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)  # (535, 3) (179, 3) (535,) (179,)

# model
model = RandomForestClassifier(criterion='entropy', n_estimators=100)
model = model.fit(train_x, train_y)

# pred 
import numpy as np
pred = model.predict(test_x)
print('예측값:', pred[:10])
print('실제값:', np.array(test_y[:10]))

# 분류 정확도
print('acc:', sum(test_y == pred) / len(test_y))
from sklearn.metrics import accuracy_score
print('acc:', accuracy_score(test_y, pred))

# 교차검증 모델 
cross_vali = cross_val_score(model, df_x, df_y, cv = 5)
print(cross_vali, ' 평균:', np.round(np.mean(cross_vali), 3))

print()
# feature로 사용할 중요 변수 확인
import matplotlib.pyplot as plt

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
