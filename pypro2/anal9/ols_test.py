# 단순 회귀 분석 : ols()
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('student.csv')
print(df.head(3))

# 상관관계 확인
print(df.corr())

# 적절성, 만족도 간의 상관관계 : 0.766853
# 위 두 변수는 인과관계가 있다 가정하고 회귀분석을 수행. x : 적절성, y : 만족도 

model = smf.ols(formula='수학 ~ 국어', data = df).fit()  # 학습된 모델이 만들어짐
print(model.summary())
# 수식 : y = 0.5705 * x + 32.1069
# t 값 : 기울기 / 표준오차(std err) = 0.5705 / 0.038
# t값을 거듭제곱하면 F-statistic의 값이 나옴

# R-squared (결정계수, 설명력) : 
# 독립변수가 종속변수의 분산을 어느정도 설명하는지를 알려준다. 
# 선형회귀 모델의 성능을 표현할 때 사용함. 절대적으로 신뢰하지는 않음.
# 15% 이상일 경우 모델을 사용한다.
# 독립변수가 하나일 때 사용(상관계수를 제곱한 값이 결정계수가 됨) 
# Adj. R-squared(수정된 결정계수) : 독립변수가 여러개일 때 사용

print('회귀 계수(intercept, slope) :', model.params)  # params를 써주면 절편과 기울기만 볼 수 있음
print('결정 계수(R squared) : ', model.rsquared)
print('p-value : ', model.pvalues)
#
# 결과 예측
print(df.국어[:5].values)
new_df = pd.DataFrame({'국어':[int(input('국어:'))]})
pred = model.predict(new_df)
print('예측결과:', pred)




 

# t값 : 두집단 간의 평균의 차이 
# 기울기 / 표준오차 
 