# 문제4) testdata/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.
# * 변수 종류 *
# satisfaction_level : 직무 만족도
# last_eval‎uation : 마지막 평가점수
# number_project : 진행 프로젝트 수
# average_monthly_hours : 월평균 근무시간
# time_spend_company : 근속년수
# work_accident : 사건사고 여부(0: 없음, 1: 있음)
# left : 이직 여부(0: 잔류, 1: 이직)
# promotion_last_5years: 최근 5년간 승진여부(0: 승진 x, 1: 승진)
# sales : 부서
# salary : 임금 수준 (low, medium, high)
#
# 조건 : Randomforest 클래스로 중요 변수를 찾고, Keras 지원 딥러닝 모델을 사용하시오.
# Randomforest 모델과 Keras 지원 모델을 작성한 후 분류 정확도를 비교하시오.

import numpy as np
import pandas as pd

data = pd.read_csv('../testdata/HR_comma_sep.csv')
print(data.head(3))
print(data.shape)  # (14999, 10)
print(data.isnull().sum())  # null값 확인

print()
print(data['sales'].unique())
# ['sales' 'accounting' 'hr' 'technical' 'support' 'management' 'IT'
# 'product_mng' 'marketing' 'RandD']

print()
# sales(부서)열 더미변수화
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
data.sales = LabelEncoder().fit_transform(data.sales)
print(data.sales.unique())  # [7 2 3 9 8 4 0 6 5 1]

print()
# salary(임금수준)열 더미변수화
data.salary = data.salary.apply(lambda x:0 if x=='low' else(1 if x =='medium' else 3))
print(data.salary.head(3))
print(data.salary.unique())  # [0 1 3]

print()
# feature / label 
x = data.loc[:, data.columns!='salary']
y = np.array(data['salary'])
print(x[:3])
print(y[:3])

print()
# train / test
from sklearn.model_selection import train_test_split, cross_val_score
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=12) 
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
# (10499, 9) (4500, 9) (10499,) (4500,)

# randomforest 모델 생성
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy', n_estimators = 500)
rf = rf.fit(train_x, train_y)

print()
# 예측
pred = rf.predict(test_x)
print('예측값:', pred[:10])
print('실제값:', test_y[:10])

# 분류 정확도
from sklearn.metrics import accuracy_score
print('acc:', accuracy_score(test_y, pred))

print('특성(변수) 중요도 :\n{}'.format(rf.feature_importances_))

# 특성 중요도 시각화
import matplotlib.pyplot as plt
def plot_feature_importances(rf):   # 특성 중요도 시각화
    n_features = x.shape[1]
    plt.barh(range(n_features), rf.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()

plot_feature_importances(rf)

# 딥러닝 model 
from keras import models, Sequential
from keras import layers
model = Sequential()
model.add(layers.Dense(units=64, activation='relu', input_shape=x.shape[1]))  
model.add(layers.Dense(units=64, activation='relu'))  
model.add(layers.Dense(units=46, activation='softmax'))  

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(train_x, train_y, epochs=15, batch_size=128, 
                    validation_split=0.2, verbose=2)

results = model.evaluate(test_x, test_y, batch_size=128, verbose=0)
print('evaluate:', results)

# 시각화 
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1) 

plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'r', label='train val_loss')
plt.xlabel('epochs')
plt.legend()
plt.show()