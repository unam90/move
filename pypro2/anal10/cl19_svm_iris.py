# 다항분류 가능 - iris dataset : 활성화 함수는 softmax 사용
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 표준화 지원 클래스
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(iris.DESCR)
print(iris.keys())

# print(iris.data)
# print(iris.target)
print()
print(iris.feature_names)

# 상관관계 확인
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3]))  # 'petal length (cm)', 'petal width (cm)'만 참여
# [[1.         0.96286543]
#  [0.96286543 1.        ]]
x = iris.data[:, [2, 3]]  # 2차원(매트릭스)
y = iris.target  # 1차원(벡터)
print(x[:4])
print(y[:4], set(y))

# train / test 분리(과적합 방지)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

"""
# scaling : 데이터 표준화 - 데이터 크기, 값의 범위(scale)를 평균 0, 분산 1이 되도록 바꿔주는 것.
# 최적화 과정에서 안정성, 수렴 속도를 향상시킬 수 있다. 과적합/과소적합 방지에도 효과적이다...
# 독립변수(feature)에 대해 작업을 한다.(종속변수는 작업 x)
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])
print()
# 표준화 원복 
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
"""

print('분류 모델 생성-------------------')
# model = LogisticRegression(C = 100.0, solver='lbfgs', multi_class='auto', random_state=0)  # 소프트맥스를 쓴다는 의미
# C 속성 : L2 규제 - 모델에 패널티 적용. 숫자 값을 조정해가며 분류 정확도를 확인.(기본값은 1.0이며 숫자값이 작을수록 더 강한 규제)

from sklearn import svm
model = svm.SVC(C=1.0)  # C값을 조정해서 과적합/과소적합을 어느정도 해결해 나갈 수 있다.
# model = svm.LinearSVC(C=1.0)  # SVC보다 속도가 개선됨
print(model)
model.fit(x_train, y_train)  # train dataset으로 모델 학습

# 분류 예측
y_pred = model.predict(x_test)  # test dataset으로 모델 검정
print('예측값:', y_pred)
print('실제값:', y_test)
print('총 갯수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))
print()
print('분류 정확도 확인 1')
print('accuracy:', accuracy_score(y_test, y_pred))  # accuracy: 0.977777
print()
print('분류 정확도 확인 2')
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측치'], colnames=['관측치'])
print(con_mat)
print('accuracy:', (con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))  # accuracy: 0.977777
print()
print('분류 정확도 확인 3')  # 주로 train, test를 비교할 때 사용한다.
print('train:', model.score(x_train, y_train))  # train: 0.9619
print('test:', model.score(x_test, y_test))     # test: 0.9777

# 모델 저장 
# ...생략

print('새로운 값으로 분류 예측')
# print(x_test[:3])
new_data = np.array([[5.1,2.4],[0.3,0.4],[3.4,0.1]])  
# 만약 표준화하고 모델을 생성했다면 new_data도 표준화 작업을 해야함.
new_pred = model.predict(new_data)
print('예측결과 :', new_pred)  # 예측결과 : [2 0 1]
# print('확률로 보기 :', model.predict_proba(new_data))

# 시각화
plt.rc('font', family='malgun gothic')      
plt.rcParams['axes.unicode_minus']= False

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')        # 점 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)       # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=model, test_idx=range(105, 150), title='scikit-learn제공') 

# SVM 모델에서 ROC 커브로 분류모형 성능 평가 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
# OneVsOneClassifier 클래스를 사용하면 이진 클래스용 모형을 OvO 방법으로 다중 클래스용 모형으로 변환한다. 
# OneVsOneClassifier 클래스는 각 클래스가 얻는 조건부 확률값을 합한 값을 decision_function으로 출력한다.

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))

y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
# 사이킷런 패키지는  roc_curve 명령을 제공한다. 
# 인수로는 타겟 y 벡터와 판별함수 벡터(혹은 확률 벡터)를 넣고 결과로는 변화되는 기준값과 그 기준값을 사용했을 때의 재현율을 반환한다.

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

# AUC(Area Under the Curve)는 ROC curve의 면적을 뜻한다. 
# 위양성률(fall out)값이 같을 때 재현률값이 클거나 재현률값이 같을 때 위양성률값이 작을수록 AUC가 1에 가까운 값이고 좋은 모형이다.

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot of a ROC curve for a specific class
plt.figure()

lw = 2

plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

from itertools import cycle
# iterable에서 요소를 반환하고 각각의 복사본을 저장하는 반복자를 만든다. 반복 가능한 요소가 모두 소모되면 저장된 사본에서 요소를 리턴한다. 
# 반복 가능한 요소가 모두 소모될때까지 무한정 반복한다.

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


 










