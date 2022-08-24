# iris dataset으로 시각화 
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline  # jupyter에서 시각화 선언. plt.show()를 안써도 됨

iris_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/iris.csv")
print(iris_data.head(3))

plt.scatter(iris_data['Sepal.Length'],iris_data['Petal.Width'])
plt.show()
print()
print(iris_data['Species'].unique())
print(set(iris_data['Species']))
cols = []

for s in iris_data['Species']:
    choice = 0
    if s == 'setosa': choice=1
    elif s == 'versicolor': choice=2
    elif s == 'virginica': choice=3
    cols.append(choice)

plt.scatter(iris_data['Sepal.Length'],iris_data['Petal.Width'], c=cols)
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Width')
plt.show()

# pandas의 시각화
from pandas.plotting import scatter_matrix
iris_col = iris_data.loc[:, 'Sepal.Length':'Petal.Width']
scatter_matrix(iris_col, diagonal = 'kde')  # kde : 밀도분포 
plt.show()

# seaborn
import seaborn as sns
sns.pairplot(iris_data, hue='Species')
plt.show()