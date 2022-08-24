# [로지스틱 분류분석 문제2] 
# 게임하는 시간, TV 시청 데이터로 안경 착용 유무를 분류하시오. (모델 평가까지)
# 안경 : 값0(착용X), 값1(착용O)
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('bodycheck.csv')
print(data.head(3), data.shape)

# dataset 분할
x = data[['게임', 'TV시청']]
y = data['안경유무']
x = x.values  # 2차원 배열
y = y.values  # 1차원 배열
print(x[:3], x.shape)
print(y[:3], y.shape)

# 모델 생성
model = LogisticRegression(C = 0.1, random_state=5)
model.fit(x, y)

print() 
y_pred = model.predict(x) 
print('예측값:', y_pred)
print('실제값:\n', y)
print()
print('정확도')
print(model.score(x, y))
print()
print('새로운 값으로 분류 예측')
print(x[:3])
new_data = np.array([[8, 9],[0, 0],[5, 6]])  
new_pred = model.predict(new_data)
print('예측결과 :', new_pred)  # 예측결과 : [1 0 1]
# print('확률로 보기 :', model.predict_proba(new_data))



