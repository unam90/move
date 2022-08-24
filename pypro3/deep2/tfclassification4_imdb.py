# 영화 리뷰 데이터로 이진 부류
# imdb : train-25000, test-25000개로 구성

from keras.datasets import imdb

(train_feature, train_label),(test_feature, test_label) = imdb.load_data(num_words=10000)
print(type(train_feature), train_feature.shape)
print(type(test_feature), test_feature.shape)
print(train_feature)  # 모든 단어에 대해 인덱싱(고유 번호)을 해서 단어 사전을 만듦
print(train_label)  # [1 0 0 ... 0 1 0]

# 참고) 원래 영문으로 변환해보기
word_index = imdb.get_word_index()
# print(word_index)  # {'fawn': 34701, 'tsukino': 52006, 'nunnery': 52007,....
print(type(word_index))  # <class 'dict'>
# print(word_index.items())  # dict_items([('fawn', 34701), ('tsukino', 52006),...

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# print(reverse_word_index)  # {34701: 'fawn', 52006: 'tsukino',....
# print(reverse_word_index.get(train_feature[0][0]))  # 0번째 단어는 the  / matrix타입이므로 [0][1] 이런 식으로 작성.
decode_review = ' '.join([reverse_word_index.get(i) for i in train_feature[0]])
print(decode_review)

# 데이터 준비 : list를 벡터화
import numpy as np

def vector_seq(datas, dim=10000):
    results = np.zeros((len(datas), dim))  # 0으로 이루어진 행렬
    # print(results)
    for i, seq in enumerate(datas):
        results[i, seq] = 1.  # # 특정 인덱스의 위치만 1로 채움
        return results

x_train = vector_seq(train_feature)
print(x_train, x_train.shape)  # (25000, 10000)
x_test = vector_seq(test_feature)
print(x_test, x_test.shape)  # (25000, 10000)
y_train = train_label  # train_label.astype('float32') 해도 되고 안해도 된다. 어차피 0아니면 1값이기 때문에
y_test = test_label

print()
# model 생성
from keras import models, layers, regularizers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,), # 10000개의 입력자료
                       kernel_regularizer=regularizers.l2(0.001)))  
# l2 규제 : 가중치 행렬의 모든 값을 제곱하고 0.001을 곱해 model 신경망의 전체 손실을 조정함. 페널티를 추가
model.add(layers.Dropout(rate=0.3)) 
# Dropout이란 과적합 방지를 위해 네트워크의 유닛의 일부만 동작하고 일부는 동작하지 않도록 하는 방법이다.
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) # 경사하강법으로 loss를 최소화

# 훈련시 validation data를 준비
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
print(len(partial_x_train), len(x_val))  # 15000 10000
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
print(len(partial_y_train), len(y_val))

# history = model.fit(x_train, y_train, ...)
history = model.fit(partial_x_train, partial_y_train, epochs=30, batch_size=512,
                    validation_data=(x_val, y_val), verbose=1)

print('evaluate:', model.evaluate(x_test, y_test, batch_size=512, verbose=0))

# 시각화
import matplotlib.pyplot as plt
history_dict = history.history
print(history_dict.keys())

loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

acc=history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='train acc')
plt.plot(epochs, val_acc, 'r', label='validation acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

pred = model.predict(x_test[:5])
print('예측값 : ', np.where(pred>0.5, 1, 0).flatten())
print('실제값 : ', y_test[:5])

# 과적합 방지 방법 - 최적화와 일반화(모델의 포용성)
# 모델의 파라미터 조정 units=?
# 가중치 규제 regularizers
# Dropout 추가
# BatchNormalization
# train / test split
# k-fold
# 훈련 데이터를 줄이기



