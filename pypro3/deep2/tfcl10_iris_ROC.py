# iris dataset으로 다항분류, 모델 성능 확인(ROC)...

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics._scorer import accuracy_scorer
from sklearn.metrics._classification import classification_report

iris = load_iris()
# print(iris.DESCR)
print(iris.keys())

x = iris.data  # feature
print(x[:2])
y = iris.target  # label
print(y[:2])
print(set(y))  # {0, 1, 2}

names = iris.target_names
feature_names = iris.feature_names
print(names)  # ['setosa' 'versicolor' 'virginica']
print(feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# label : one-hot 
# 종류 : sklearn - OneHotEncoder, keras - to_categorical, numpy - np.eye, pandas - pd.get_dummies
print(y.shape)  # (150,)
onehot = OneHotEncoder(categories='auto') 
y = onehot.fit_transform(y[:, np.newaxis]).toarray()
print(y.shape)  # (150, 3)
print(y[:2])    # [[1. 0. 0.] ...

# feature : 표준화 / 정규화가 필요하다면 권장
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])

# train_test split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (105, 4) (45, 4) (105, 3) (45, 3)

N_FEATURES = x_train.shape[1]  # 4  / 상수값으로 담을 때는 대문자를 써준다
N_CLASSES = y_train.shape[1]   # 3

# model 
from keras.models import Sequential
from keras.layers import Dense

# 노드(뉴런)의 갯수를 변경해 가며 모델 작성 함수 
def create_custom_model_func(input_dim, output_dim, out_nodes, n, model_name='model'):
    # print(input_dim, output_dim, out_nodes, n, model_name)
    def create_model():
        model = Sequential(name=model_name)
        for _ in range(n):
            model.add(Dense(units=out_nodes, input_dim=input_dim, activation='relu'))
        
        model.add(Dense(units=output_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model     # closure
    return create_model  # closure
    
models = [create_custom_model_func(N_FEATURES, N_CLASSES, 10, n, 'model_{}'.format(n)) for n in range(1, 4)]
print(len(models))  # 3

for cre_model in models:
    print('-----')
    cre_model().summary()

print()    
history_dict = {}

for cre_model in models:
    model = cre_model() 
    print('모델명:', model.name)
    histories = model.fit(x_train, y_train, batch_size=5, epochs=50, 
                          verbose=0, validation_split=0.3)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('test loss:', scores[0])
    print('test acc:', scores[1])
    history_dict[model.name] = [histories, model]  # dict 타입으로 key : histories, value : model 담기
    
print(history_dict)

# 시각화 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

for model_name in history_dict:
    print('h_d:', history_dict[model_name][0].history['acc'])  
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)

    ax1.set_ylabel('val acc')
    ax2.set_ylabel('val loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
plt.show()

# ROC curve : 모델 성능을 평가
plt.figure()
plt.plot([0,1], [0,1], 'k--')

from sklearn.metrics import roc_curve, auc

for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    plt.plot(fpr, tpr, label='{}, AUC value:{:.3f}'.format(model_name, auc(fpr, tpr)))
    
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

print('------k-fold(교차검증) 수행 - 3개 모델 전체를 대상으로 진행-------')
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
create_model = create_custom_model_func(N_FEATURES, N_CLASSES, 10, 3)

estimator = KerasClassifier(build_fn = create_model, epochs=50, batch_size=10, verbose=2)
scores = cross_val_score(estimator, x_scale, y, cv=10)  # cv : 몇 겹으로 할 것인지
print('acc:{:0.2f}(+/-{:0.2f})'.format(scores.mean(), scores.std()))
# acc:0.84(+/-0.15) : 레이어 1
# acc:0.94(+/-0.06) : 레이어 2
# acc:0.95(+/-0.06) : 레이어 3

# 다양한 방법으로 모델 성능을 확인한 후 가장 우수한 모델이 되는 파라미터를 사용해서 정식 모델을 만듦
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# ModelCheckpoint, EarlyStopping 적용. 지금은 생략
model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=2)
print('evaluate:', model.evaluate(x_test, y_test))
pred = np.argmax(model.predict(x_test), axis=1)
print('예측값:', pred)
real_y = np.argmax(y_test, axis=1).reshape(-1, 1)  # [[1 0 0] ==> 0
print('실제값:', real_y.ravel())
print('분류 실패 수 :', (pred != real_y.ravel()).sum())

print('confusion matrix---')
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(real_y, pred))
print(accuracy_score(real_y, pred))
print(classification_report(real_y, pred))

# 새로운 값으로 분류 
new_x = [[5.1, 3.5, 1.1, 2.2], [3.3, 3.3, 3.3, 3.3], [7.0, 6.0, 5.0, 3.0]]
new_x = StandardScaler().fit_transform(new_x)
new_pred = model.predict(new_x)
print('예측결과:', np.argmax(new_pred, axis=1).reshape(-1, 1).flatten())  # 예측결과: [0 2 2]






