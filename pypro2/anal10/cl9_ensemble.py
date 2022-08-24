# 앙상블 개념의 핵심 개념은 다양한 종류의 여러 estimator를 결합하여 더 좋은 estimator를 만드는 것이다. 
# 앙상블 방법의 종류는 estimator들을 어떻게 결합할 것인지에 의해 결정된다. 
# 가장 단순한 앙상블 방법으로는 투표를 기반으로 base estimator들을 결합하여 최종 estimator를 만드는 것이다.
# LogisticRegression + DecisioTreeClassifier + NkeighborsClassifier 로 보팅 분류기 생성
# data는 사이킷런의 유방암 진단 데이터, breast_cancer를 사용

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

cancer = load_breast_cancer()
print(cancer.keys())
data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
print(data_df.head(3))

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, \
                                                    test_size=0.2, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(set(y_train))  # label이 0이면 malignant 양성, 1이면 benign 음성이다.

# 개별 모델 작성
logi_model = LogisticRegression()
knn_model = KNeighborsClassifier(n_neighbors = 3)
dec_model = DecisionTreeClassifier()

voting_model = VotingClassifier(estimators=[('LR', logi_model),('KNN', knn_model),('Decision', dec_model)],
                                voting='soft')  # 'hard'가 기본

classifiers = [logi_model, knn_model, dec_model]

# 개별 모델의 학습 및 예측 평가
for cl in classifiers:
    cl.fit(x_train, y_train) 
    pred = cl.predict(x_test)
    class_name = cl.__class__.__name__
    print('{0} 정확도 : {1:.4f}'.format(class_name, accuracy_score(y_test, pred)))
    
# 앙상블 모델의 학습 및 예측 평가
voting_model.fit(x_train, y_train)
vpred = voting_model.predict(x_test)
print(vpred)
print('{0} 정확도 : {1:.4f}'.format('보팅 분류기', accuracy_score(y_test, vpred)))


