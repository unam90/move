# K-nn 단순 실습
from sklearn.neighbors import KNeighborsClassifier
from networkx.algorithms.link_prediction import common_neighbor_centrality

kmodel = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')

train =[
    [5,3,2],
    [1,3,5],
    [4,5,7]
]

print(train)
label = [0, 1, 1]

import matplotlib.pyplot as plt
# plt.plot(train, 'o')
# plt.xlim([-1, 5])
# plt.ylim([0, 10])
# plt.show()

kmodel.fit(train, label)
pred = kmodel.predict(train)

print('예측값:', pred)
print('정확도:{:.2f}'.format(kmodel.score(train, label)))

new_data = [[1,2,8], [6,4,1]]
new_pred = kmodel.predict(new_data)
print('예측 결과:', new_pred)

print('\n---최적의 k값 알아내기--------------')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                stratify=cancer.target, random_state=66)

# stratify : class 비율을 train/validation에 유지. 쏠림 방지

print(x_train[:2])
print(y_train[:2])

train_accuracy = []
test_accuracy = []
neighbor_set = range(1, 10)

for n_neighbor in neighbor_set:
    clf = KNeighborsClassifier(n_neighbors=n_neighbor).fit(x_train, y_train)
    train_accuracy.append(clf.score(x_train, y_train))
    test_accuracy.append(clf.score(x_test, y_test))

import numpy as np
print('train 분류 평균 정확도:', np.mean(train_accuracy))    
print('test 분류 평균 정확도:', np.mean(test_accuracy))    

plt.plot(neighbor_set, train_accuracy, label='훈련 정확도')
plt.plot(neighbor_set, test_accuracy, label='검증 정확도')
plt.xlabel('k값')
plt.ylabel('분류 정확도')
plt.legend()
plt.show()











