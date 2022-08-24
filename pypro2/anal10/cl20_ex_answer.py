import pandas as pd 
from sklearn import svm, metrics
from sklearn.model_selection._split import train_test_split

# 데이터 가공 
heartdata = pd.read_csv("../testdata/Heart.csv")
print(heartdata.info())

data = heartdata.drop(["ChestPain", "Thal"], axis = 1) # object type은 제외
data.loc[data.AHD=="Yes", 'AHD'] = 1
data.loc[data.AHD=="No", 'AHD'] = 0
print(heartdata.isnull().sum())      # Ca 열에 결측치 4개

Heart = data.fillna(data.mean())   # CA에 결측치는 평균으로 대체
label = Heart["AHD"]
features = Heart.drop(["AHD"], axis = 1)

x_train, x_test, y_train, y_test = \
    train_test_split(features, label, test_size = 0.3, random_state = 12)

model = svm.SVC(C=0.1).fit(x_train, y_train)

# 예측 
import numpy as np
pred = model.predict(x_test)
print('예측값 : ', pred)
print('실제값 : ', np.array(y_test))

# 분류 정확도 
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print('분류 정확도 : ', metrics.accuracy_score(y_test, pred))

# 예측
new_test = x_test[:2].copy()
print(new_test)
new_test['Age'] = 10
new_test['Sex'] = 0
print(new_test)
new_pred = model.predict(new_test)
print('예측결과 : ', new_pred)

