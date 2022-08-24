# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
#  - 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#  - 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#     참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 
# 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  

from scipy import stats 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

data = StringIO("""
구분,지상파,종편,운동
1,0.9,0.7,4.2
2,1.2,1.0,3.8
3,1.2,1.3,3.5
4,1.9,2.0,4.0
5,3.3,3.9,2.5
6,4.1,3.9,2.0
7,5.8,4.1,1.3
8,2.8,2.1,2.4
9,3.8,3.1,1.3
10,4.8,3.1,35.0
11,NaN,3.5,4.0
12,0.9,0.7,4.2
13,3.0,2.0,1.8
14,2.2,1.5,3.5
15,2.0,2.0,3.5
""") 
print(data)

df = pd.read_csv(data)
print(df.head(3), df.shape)

# 결측치는 평균값으로 대체 
avg = df['지상파'].mean()
df = df.fillna(avg)

print(df)

# 운동 10시간 초과는 이상치로 한다. 제거 
for d in df.운동:
    if d > 10:
        df = df[df.운동 != d]

print(df)

# 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
x = df.지상파
y = df.운동
model = stats.linregress(x, y)
print('기울기:', model.slope)
print('절편:', model.intercept)
fit_value = np.polyval([model.slope, model.intercept], df.지상파)  # 예측값 구하기
print('fit_value:', fit_value)

plt.scatter(df.지상파, df.운동)
plt.plot(df.지상파, fit_value, 'r-')
plt.show()