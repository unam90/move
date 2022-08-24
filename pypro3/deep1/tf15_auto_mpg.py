# 다중선형회귀분석 : 자동차 연비 예측

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers

pd.set_option('display.max_columns', None)
dataset = pd.read_csv('../testdata/auto-mpg.csv', na_values= '?')
print(dataset.head(2))
# print(dataset.describe())
dataset = dataset.dropna(axis=0)  # 결측치 제거 
# print(dataset.describe())
del dataset['car name']  # 필요없는 car name 칼럼 제거

print(dataset.head(2))
print(dataset.corr()['mpg'])  # mpg칼럼을 기준으로 상관관계 확인
      
# 시각화
# sns.pairplot(dataset[['mpg', 'weight', 'horsepower', 'displacement']], diag_kind='kde')
# plt.show()

print()
# train/test
print(dataset.shape)  # (392, 8)
train_dataset = dataset.sample(frac=0.7, random_state=123)  # frac: 비율
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset.shape)  # (274, 8)
print(test_dataset.shape)   # (118, 8)

# 표준화 작업 : 수식을 사용 : (요소값 - 평균) / 표준편차
train_stat = train_dataset.describe()
train_stat.pop('mpg')  # mpg열은 label로 사용하므로 제외
# print(train_stat)

train_stat = train_stat.transpose()
print(train_stat)

# mpg
train_labels = train_dataset.pop('mpg')  # mpg를 빼냄
print(train_labels[:2])
test_labels = test_dataset.pop('mpg')
print(test_labels[:2])

# 표준화 시키는 함수 만들기
def std_func(x):
    return(x - train_stat['mean'] / train_stat['std'])

# print(std_func(10))
print(train_dataset[:2])
print(std_func(train_dataset[:2]))

# 표준화된 feature dataset 
st_train_data = std_func(train_dataset)
st_test_data = std_func(test_dataset)

print('------model 만들기-------')
from keras.models import Sequential
from keras.layers import Dense

def build_model():
    network = Sequential([
        Dense(units=64, activation='relu', input_shape=[7]),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # activation='linear' 는 생략가능
    ])
    # opti = tf.keras.optimizers.RMSprop(0.001)
    opti = tf.keras.optimizers.Adam(0.001)
    network.compile(optimizer=opti, loss='mean_squared_error', metrics=['mean_absolute_error','mean_squared_error'])
    
    return network

model = build_model()
print(model.summary())

# fit하기 전에 모델로 predict 가능(결과는 신경X)
print(model.predict(st_train_data[:1]))

# fit
epochs = 1000

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5)  # 5회가 지나도록 loss가 안떨어지면 멈춘다. 

history = model.fit(st_train_data, train_labels, batch_size=32,
                    epochs=epochs, validation_split=0.2, verbose=2, callbacks=[es])

df = pd.DataFrame(history.history)
print(df.head(3))

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))
  
  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
  plt.legend()
  
  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
  plt.legend()
  plt.show()

plot_history(history)

# evaluate
loss, mae, mse = model.evaluate(st_test_data, test_labels)
print('loss:{:5.3f}'.format(loss))
print('mae:{:5.3f}'.format(mae))
print('mse:{:5.3f}'.format(mse))

# predict 
test_pred = model.predict(st_test_data).flatten()
print('예측값:', test_pred)
print('실제값:', test_labels.values)






