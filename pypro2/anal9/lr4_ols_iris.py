# 단순선형회귀 분석모델 : iris dataset을 사용
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns 
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(iris.head(3), type(iris), iris.shape)

# 상관관계 확인
print(iris.corr(method='pearson'))

print('연습1 : 상관관계가 약한 변수를 사용해 선형회귀모델 작성')
# sepal_width, sepal_length : -0.117570 약한 음의 상관관계
result1 = smf.ols(formula='sepal_length ~ sepal_width', data = iris).fit()
print('요약결과1 : ', result1.summary())
# Prob (F-statistic): p-value 모델의 유의함을 확인하는 지표 0.152 > 유의수준(알파) 0.05
# 이므로 모델로 적합하지 않다.
print('R-squared:', result1.rsquared)
print('p-value:', result1.pvalues[1])

plt.scatter(iris.sepal_width, iris.sepal_length)
plt.plot(iris.sepal_width, result1.predict(), color='r')
plt.show()

print()
print('연습2 : 상관관계가 강한 변수를 사용해 선형회귀모델 작성')
# petal_length, sepal_length : 0.871754 양의 상관관계가 매우 강하다.

result2 = smf.ols(formula='sepal_length ~ petal_length', data = iris).fit()
print('요약결과2 : ', result2.summary())
# Prob (F-statistic): p-value 모델의 유의함을 확인하는 지표 1.04e-47 < 유의수준(알파) 0.05
# 이므로 모델로 적합하다.
print('R-squared:', result2.rsquared)  # 0.75995
print('p-value:', result2.pvalues[1])  # 1.0386674194499307e-47

print('실제값:', iris.sepal_length[:5].values)
print('예측값:', result2.predict()[:5])

plt.scatter(iris.petal_length, iris.sepal_length)
plt.plot(iris.petal_length, result2.predict(), color='r')
plt.show()

# 새로운 값(petal_length)으로 sepal_length를 예측
# petal length : 1.1의 sepal_length 예측값? 4.756390000000001
print('수식 사용:', 0.4089 * 1.1 + 4.3066)  # y = ws + b 

# 함수로 예측
new_data = pd.DataFrame({'petal_length':[1.1, 0.5, 5.0]})
y_pred = result2.predict(new_data)
print('예측 결과는 ', y_pred.values)

print('독립변수가 복수 : 다중회귀분석')
result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width + sepal_width', 
                  data = iris).fit()
print(result3.summary())

