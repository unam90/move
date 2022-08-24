from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("testdata/winequality-red.csv")
df_x = df.drop(['quality'], axis=1)  # feature로 사용. quality를 제외한 나머지 열
df_y = df['quality']  # label로 사용
print(df_y.unique())  # [5 6 7 4 8 3]
print(df_x.columns)  # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', ...

# 이하 소스 코드를 적으시오.
# 1) train_test_split (7:3), random_state=12
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=12)


# 2) DecisionTreeClassifier 클래스를 사용해 분류 모델 작성 (criterion='entropy', n_estimators=500)
model = RandomForestClassifier(criterion='entropy', n_estimators=500)

# 3) 분류 정확도 출력
fit = model.fit(df_x_train, df_y_train)
print('분류 정확도 확인')
print('train:', model.score(df_x_train, df_y_train)) 
print('test:', model.score(df_x_test, df_y_test))