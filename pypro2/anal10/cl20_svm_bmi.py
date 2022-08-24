# BMI 식을 이용하여 데이터 만들기
# BMI를 이용한 비만도 계산은 자신의 몸무게를 키의 제곱으로 나누는 것으로 공식은 kg/㎡.
# 예) 키: 170, 몸무게: 68     68 / ((170 / 100) * (170 / 100))
print(68 / ((170 / 100) * (170 / 100)))  # 23.5294

"""
import random

def calc_bmi(h, w):
    bmi = w / (h/100) ** 2
    if bmi < 18.5: return 'thin'
    if bmi < 25.0: return 'normal'
    return 'fat'

# print(calc_bmi(190, 78))

fp = open('bmi.csv', 'w')
fp.write('height,weight,label\n')  # 제목


# 무작위로 데이터 생성
cnt = {'thin':0, 'normal':0, 'fat':0}

random.seed(12)
for i in range(50000):
    h = random.randint(150, 200)  # 150cm~200cm까지
    w = random.randint(35, 100)   # 35kg~100kg까지
    label = calc_bmi(h,w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h,w,label))

fp.close() 
"""

# SVM으로 분류모델 만들기
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv('bmi.csv')
print(tbl.head(3), tbl.shape)
print(tbl.info())

label = tbl['label']
print(label[:3])

w = tbl['weight']/100  # 정규화
print(w[:3].values)
h = tbl['height']/200  # 정규화
print(h[:3].values)

wh = pd.concat([w, h], axis=1)
print(wh.head(3), wh.shape)

# label을 dummy화 
label = label.map({'thin':0, 'normal':1, 'fat':2})
print(label[:3])

# train, test로 나누기
data_train, data_test, label_train, label_test = train_test_split(wh, label, test_size=0.3, random_state=1)
print(data_train.shape, data_test.shape, label_train.shape, label_test.shape)
# (35000, 2) (15000, 2) (35000,) (15000,)  

print()
# model 
model = svm.SVC(C=0.01).fit(data_train, label_train)
# print(model)

pred = model.predict(data_test)
print('예측값:', pred[:10])
print('실제값:', label_test[:10].values)

# accuracy
acc_score = metrics.accuracy_score(label_test, pred)
print('정확도:', acc_score)  # 정확도: 0.9705

print()
# 교차 검증 모델
from sklearn import model_selection
cross_vali = model_selection.cross_val_score(model, wh, label, cv=3)
print('각각의 검증 정확도:', cross_vali)  # 각각의 검증 정확도: [0.96940061 0.96586068 0.96681867]
print('평균 검증 정확도:', cross_vali.mean())  # 평균 검증 정확도: 0.9673599891736715

# 시각화 
tbl2 = pd.read_csv('bmi.csv', index_col=2)
def scatter_func(lbl, color):
    b = tbl2.loc[lbl]
    plt.scatter(b['weight'], b['height'], c=color, label=lbl)
    
scatter_func('fat', 'red')
scatter_func('normal', 'yellow')
scatter_func('thin', 'blue')
plt.legend()
plt.show()

