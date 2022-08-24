# [분류분석 문제2] 
# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
# 안경 : 값1(착용X), 값2(착용O)
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

data = pd.read_csv("../testdata/bodycheck.csv")
print(data.head(2))

# dataset 분할 
x = data[["게임","TV시청"]]
y = data["안경유무"]
x = x.values   # 2차원 배열
y = y.values   # 1차원 배열
print(x[:3], x.shape)
print(y[:3], y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)   
print(x_train.shape, x_test.shape)   # (14, 2) (6, 2)

"""
# feature 스케일링
sc = StandardScaler()
sc.fit(x_train,x_test)           
x_train = sc.transform(x_train)      
x_test = sc.transform(x_test) 
"""

# model 생성 하기
model = LogisticRegression(C=0.1, random_state=12) 
print(model)
model.fit(x_train, y_train) 
 
# model 검정
y_pred = model.predict(x_test)      
print(y_pred)  
print("예측값: ", y_pred)
print("실제값: ", y_test)
print("총 갯수: %d, 오류 수: %d"%(len(y_test),(y_test != y_pred).sum()))

print("정확도1 : %.3f" %accuracy_score(y_test,y_pred))

con_mat = pd.crosstab(y_test, y_pred)
print(con_mat)
print("정확도2", (con_mat.iloc[0,0] + con_mat.iloc[1,1])/len(y_test)) 

print("정확도3", model.score(x_test, y_test))

# 입력 받은 새로운 값으로 예측하기
tmp1 = int(input("게임 시간 입력(정수):"))  
tmp2 = int(input("TV 시청 시간 입력(정수):")) 

new_data = [[tmp1, tmp2]]
pred = model.predict(new_data)
print(pred)

result = "안경착용 X" if pred == 0 else "안경착용 O"

print("게임시간이 %d이고 TV시청시간이 %d인 경우 %s"%(tmp1, tmp2, result))   
