# 회귀분석 문제 5) 
# Kaggle 지원 dataset으로 회귀분석 모델(LinearRegression)을 작성하시오.
# testdata 폴더 : Consumo_cerveja.csv
#
# Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
# feature : Temperatura Media (C) : 평균 기온(C)
#           Precipitacao (mm) : 강수(mm)
# label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오
#
# 조건 : NaN이 있는 경우 삭제!

import pandas as pd

data = pd.read_csv("testdata/Consumo_cerveja.csv")
print(data.head(2), len(data))  # 941 행
data = data.dropna(axis=0)
print(data.isnull().sum())  # 결측치 확인

data = data.iloc[:, [1,4,6]] # 필요한 열만 추출
data.columns = ['평균기온','강수량','맥주소비량']
print(data.tail(2), len(data))  # 365 행

# 기온, 강수량 데이터 수정
data['평균기온'] = data['평균기온'].str.replace(",", ".")
data['강수량'] = data['강수량'].str.replace(",", ".")

print(data.info())   # object -> float

data = data.astype({'평균기온':'float'})
data = data.astype({'강수량':'float'})
print(data.info())

# 모델
from sklearn.linear_model import LinearRegression
x = data[['평균기온','강수량']]   # feature는 2차원 배열
y = data['맥주소비량']   # label은 1차원 배열
model = LinearRegression().fit(x, y)
print('기울기 : ', model.coef_)
print('절편 : ', model.intercept_)

# 예측
pred = model.predict(x)
print('예측값 : ', pred[:5].flatten())
print('실제값 : ', y[:5].values)

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y, pred))

print()
# 새로운 값으로 예측
print(data[:2])
new_x = [[11.0, 0.0],[38.0, 20.0]]
new_pred = model.predict(new_x)
print('새로운 값 예측 결과 : ', new_pred.flatten())  # 차원 축소함
print('새로운 값 예측 결과 : ', new_pred.ravel())    # 차원 축소함
import numpy as np
print('새로운 값 예측 결과 : ', np.squeeze(new_pred)) # 차원 축소함

