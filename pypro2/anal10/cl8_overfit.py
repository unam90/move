# 분류 모델의 과적합 방지 처리 방법
# train-test split, KFold, GridSearchCV
# iris dataset을 사용
# 모델은 DecisionTreeClassifier 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris.keys())

train_data = iris.data 
train_label = iris.target
print(train_data[:3], train_data.shape)
print(train_label[:3], train_label.shape)

# 분류 모델
dt_clf = DecisionTreeClassifier()  # 얘 이외에 다른 분류 모델 클래스를 사용해도 된다.
print(dt_clf)

dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(train_data)
print('예측값:', pred)
print('실제값:', train_label)
print('분류 정확도:', accuracy_score(train_label, pred))  # 과적합 발생...

print('과적합 방지 처리 방법1 : 학습데이터와 검정데이터로 나누어 모델을 평가')
# train / test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=121)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
dt_clf.fit(x_train, y_train)    # train data로 학습
pred2 = dt_clf.predict(x_test)  # test data로 검정
print('예측값:', pred2)
print('실제값:', y_test) 
print('분류 정확도:', accuracy_score(y_test, pred2))  # 분류 정확도: 0.95555 포용성을 위한 모델 완성
# 예) 꼬리가 없는 댕댕이도 댕댕이로 분류할 수 있어야 한다.

print('과적합 방지 처리 방법2 : 교차검증(cross validation, K-Fold)-----')
# 학습데이터를 쪼개 모델 학습시 학습과 평가를 병행 
# k개의 data fold set을 만들어 k번 만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행
from sklearn.model_selection import KFold
import numpy as np
features = iris.data
label = iris.target

dt_clf = DecisionTreeClassifier(criterion='entropy', random_state=123)
kfold = KFold(n_splits=5)  # 5겹
cv_acc = []
print('iris shape:', features.shape)  # (150, 4)
# 전체 행 수가 150, 학습데이터 : 4/5(120개), 검증데이터 : 1/5(30개)로 분할해서 학습함

n_iter = 0
for train_index, test_index in kfold.split(features):
    # print('n_iter:', n_iter)
    # print('train_index', len(train_index))
    # print('test_index', len(test_index))
    # n_iter += 1
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # 학습 및 평가 
    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test)
    n_iter += 1
    # 학습 할 때마다 정확도 측정
    acc = np.round(accuracy_score(y_test, pred),3)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('반복수:{0}, 정확도:{1}, 학습데이터 크기:{2}, 검증데이터 크기:{3}'.format(n_iter, acc, train_size, test_size))
    print('반복수:{0}, 검증데이터 인덱스:{1}'.format(n_iter, test_index))
    cv_acc.append(acc)  # 정확도를 list에 저장
    
print('평균 검증 정확도:', np.mean(cv_acc))  # 평균 검증 정확도: 0.9199

print('과적합 방지 처리 방법2-1 : 불균등한 분포를 가진 레이블 데이터에 대한 교차검증(cross validation, K-Fold)-----')
from sklearn.model_selection import StratifiedKFold
# 예) 편향, 왜곡된 대출 사기 데이터 : 대부분 정상, 극히 일부만 사기, 이메일(극히 일부만 스팸), 강우(일부만 비가 옴) 
cv_acc = []
n_iter = 0
skfold = StratifiedKFold(n_splits=5)

for train_index, test_index in skfold.split(features, label):
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # 학습 및 평가 
    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test)
    n_iter += 1
    # 학습 할 때마다 정확도 측정
    acc = np.round(accuracy_score(y_test, pred),3)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('반복수:{0}, 정확도:{1}, 학습데이터 크기:{2}, 검증데이터 크기:{3}'.format(n_iter, acc, train_size, test_size))
    print('반복수:{0}, 검증데이터 인덱스:{1}'.format(n_iter, test_index))
    cv_acc.append(acc)  # 정확도를 list에 저장
    
print('평균 검증 정확도:', np.mean(cv_acc))  # 평균 검증 정확도: 0.9534
print()
print('과적합 방지 처리 방법2-2 : 교차검증 쉽게하기 -----')
from sklearn.model_selection import cross_val_score
data = iris.data
label = iris.target

score = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=5)  # K-fold 5겹이 된다.
print('평균 검증 정확도:', np.round(np.mean(score), 3))  # 평균 검증 정확도: 0.953
print()
print('과적합 방지 처리 방법3 : 교차검증과 하이퍼 파라미터(최적의 속성) 튜닝을 한꺼번에 하기-----')
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}  # dict type

grid_dtree = GridSearchCV(dt_clf, param_grid=parameters, cv=5, refit=True)

grid_dtree.fit(x_train, y_train)  # 자동으로 복수 개의 내부 모형을 생성하고 최적의 파라미터를 제시해준다.

# GridSearchCV 결과를 DataFrame으로 변환
import pandas as pd
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df)
print('GridSearchCV 최적의 속성:', grid_dtree.best_params_) 
# GridSearchCV 최적의 속성: {'max_depth': 3, 'min_samples_split': 2}
print('GridSearchCV 최고 정확도:', grid_dtree.best_score_)  # 0.94166

estimatorModel = grid_dtree.best_estimator_   # 최적의 분류 모델 생성
pred = estimatorModel.predict(x_test)
print('pred:', pred)
print('테스트 데이터에 의한 분류 정확도:', accuracy_score(y_test, pred))

