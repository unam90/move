# sk-learn 라이브러리가 제공하는 몇가지 Regression 클래스 사용
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR

adver = pd.read_csv('Advertising.csv', usecols=[1,2,3,4])
print(adver.head(2))

print(adver.corr(method='pearson'))

x = np.array(adver.loc[:, 'tv':'newspaper'])
y = np.array(adver.sales)
print(x[:2])
print(y[:2])

print('LinearRegression--------------')
lmodel = LinearRegression().fit(x, y)
lpred = lmodel.predict(x)
print('LinearRegression pred:', lpred[:5])
print('real:', y[:5])
print('k_r2:', r2_score(y, lpred))  # 0.89721
print()
print('RandomForestRegressor--------------')
rmodel = RandomForestRegressor(criterion='squared_error').fit(x, y)
rpred = rmodel.predict(x)
print('RandomForestRegressor pred:', rpred[:5])
print('real:', y[:5])
print('r_r2:', r2_score(y, rpred))  # 0.9973
print()
print('XGBRegressor--------------')
xmodel = XGBRegressor(criterion='squared_error').fit(x, y)
xpred = xmodel.predict(x)
print('XGBRegressor pred:', xpred[:5])
print('real:', y[:5])
print('x_r2:', r2_score(y, xpred))  # 0.9999
print()
print('KNeighborsRegressor--------------')
kmodel = KNeighborsRegressor(n_neighbors=3).fit(x, y)
kpred = kmodel.predict(x)
print('KNeighborsRegressor pred:', kpred[:5])
print('real:', y[:5])
print('k_r2:', r2_score(y, kpred))  # 0.9680
print()
print('SVM--------------')
smodel = SVR().fit(x, y)
spred = smodel.predict(x)
print('SVM pred:', spred[:5])
print('real:', y[:5])
print('s_r2:', r2_score(y, spred))  # 0.8896

