# 차트 종류 경험하기
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family = 'malgun gothic')  
plt.rcParams['axes.unicode_minus'] = False

x = np.arange(10)
y = np.sin(x)
z = np.cos(x)

"""
# 차트 영역 객체 선언 방법1
plt.figure()  # matplotlib 스타일의 인터페이스
plt.subplot(2,1,1)  # row, column, panel number
plt.plot(x, y)
plt.subplot(2,1,2)
plt.plot(x, z)
plt.show()

# 차트 영역 객체 선언 방법 1-1
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)  # 1행 2열
ax2 = fig.add_subplot(1,2,2)
ax1.hist(np.random.randn(10), bins=5, alpha=0.3)  # bins(구간수)
ax2.plot(np.random.randn(10))
plt.show()

# 차트 영역 객체 선언 방법2
fig, ax = plt.subplots(nrows=2, ncols=1)  # 객체지향 인터페이스
ax[0].plot(x, np.sin(x))  # 첫번째 행
ax[1].plot(x, np.cos(x))  # 두번째 행
plt.show()
"""
 
data = [50,80,100,70,90]
"""
# plt.bar(range(len(data)), data)  # bar : 세로 막대 그래프
plt.barh(range(len(data)), data)  # barh : 가로 막대 그래프
plt.show()


plt.pie(data, explode=(0,0.1,0,0,0.5), colors=['yellow','red','blue'])  # 원형 그래프
plt.show()


plt.boxplot(data)  # 박스 그래프
plt.show()


plt.scatter(data, data)  # 산점도 
plt.show()


# DataFrame 자료로 그래프 그리기
import pandas as pd
fdata = pd.DataFrame(np.random.randn(1000, 4), # 1000행 4열
                     index=pd.date_range('1/1/2000', periods=1000),
                     columns=list('ABCD')) 
plt.plot(fdata.cumsum())
plt.plot(fdata)
plt.show()

# pandas의 plot기능
print(type(fdata))  # class 'pandas.core.frame.DataFrame'>
fdata.plot(kind='box')
plt.xlabel('time')
plt.ylabel('data')
plt.show()

#...이외에도 종류가 엄청 많다.
"""

# seaborn 모듈 : matplotlib의 기능을 추가한 시각화 패키지
import seaborn as sns

titanic = sns.load_dataset('titanic')
print(titanic.info())

# sns.displot(titanic['age'])
# sns.boxplot(y='age', data=titanic)
t_pivot = titanic.pivot_table(index='class', columns='sex', aggfunc='size')
sns.heatmap(t_pivot)  # 밀도가 높을수록 색깔이 짙음
plt.show()



