# 회귀분석 문제 5) 
# Kaggle 지원 dataset으로 회귀분석 모델(LinearRegression)을 작성하시오.
# testdata 폴더 : Consumo_cerveja.csv
#
# Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
# feature : Temperatura Media (C) : 평균 기온(C)
#             Precipitacao (mm) : 강수(mm)
# label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오
# 조건 : NaN이 있는 경우 삭제

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api

data = pd.read_csv('Consumo_cerveja.csv', usecols=[1,4,6])
print(data.head(3), len(data))
print(data.info())
data = data.dropna(axis=0)  # NaN이 있는 경우 행 지우기 
print(data.isnull().sum())  # 결측치 확인

# 칼럼명 변경
data.columns = ['평균기온', '강수량', '맥주소비량']
print(data.head(3))

# 평균기온, 강수량 데이터 수정
data['평균기온'] = data['평균기온'].str.replace(',','.')
data['강수량'] = data['강수량'].str.replace(',','.')

# 평균기온, 강수량 데이터 타입 수정 object -> float
data = data.astype({'평균기온':'float'})
data = data.astype({'강수량':'float'})
print(data.info())

# 모델 만들기
x = data[['평균기온', '강수량']].values
print(x[:3], x.shape)
y = data['맥주소비량'].values
print(y[:3], y.shape)
print()
lmodel = LinearRegression().fit(x, y)
print('기울기:', lmodel.coef_, '절편:', lmodel.intercept_)

# 예측
pred = lmodel.predict(x)
print('예측값:', np.round(pred[:10],1))
print('실제값:', y[:10])

print('설명력:', r2_score(y, pred))

print()
# 새로운 값으로 예측
new_data = [[10, 50]]
new_pred = lmodel.predict(new_data)
print('기온 %s도, 강수량 %s 인 경우, 맥주소비량은 약 %s입니다.'%(new_data[0][0], new_data[0][1], round(new_pred[0], 1)))