import pandas as pd
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn import tree

data = pd.read_csv('testdata/titanic_data.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age','Fare'])
print(data.head(2), data.shape) # (891, 12)
data.loc[data["Sex"] == "male","Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1
print(data["Sex"].head(2))
print(data.columns)

feature = data[["Pclass", "Sex", "Fare"]]
label = data["Survived"]

# train / test 
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size = 0.3, random_state = 12)

# DecisionTree 분류 모델 작성
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# print(model)
print()
# 분류 정확도 출력
fit = model.fit(feature_train, label_train)
print('분류 정확도 확인')
print('train:', model.score(feature_train, label_train)) 
print('test:', model.score(feature_test, label_test))


x = [1,2,3,4,5]
y = [8,7,6,4,5]
print('상관계수:', np.corrcoef(x, y)[0, 1])





