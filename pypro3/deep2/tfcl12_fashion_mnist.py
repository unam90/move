# Fashion Mnist dataset

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# label name
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

f_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = f_mnist.load_data()
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot']
print(set(train_labels))  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# plt.imshow(train_images[0], cmap='gray')
# plt.show()

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)  # 5행 5열
#     plt.xticks([])  # tick 안보이게
#     plt.yticks([])
#     plt.xlabel(class_names[train_labels[i]])
#     plt.imshow(train_images[i], cmap='gray')
# plt.show()  
 
# 정규화 
train_images = train_images / 255.0
test_images = test_images / 255.0

# model 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    keras.layers.Dense(units=64, activation=tf.nn.relu),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=128, epochs=30, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('모델 성능 loss:', test_loss)
print('모델 성능 acc:', test_acc)

pred = model.predict(test_images)
print(pred[0])
print('예측값:', np.argmax(pred[0]))
print('실제값:', test_labels[0])

# 시각화
def plot_image_func(i, pred_arr, true_label, img):
    pred_arr, true_label, img = pred_arr[i], true_label[i], img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    
    pred_label = np.argmax(pred_arr)
    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[pred_label], 100*np.max(pred_arr), 
                                         class_names[true_label]), color = color)
    
def plot_value_func(i, pred_arr, true_label):
    pred_arr, true_label = pred_arr[i], true_label[i]
    thisPlot = plt.bar(range(10), pred_arr)
    plt.ylim([0, 1])
    pred_label = np.argmax(pred_arr)
    thisPlot[pred_label].set_color('red')   # 예측값
    thisPlot[pred_label].set_color('blue')  # 실제값
    
    
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)  # 1행 2열 중에 1열은 이미지 출력
plot_image_func(i, pred, test_labels, test_images)

plt.subplot(1,2,2)  # 1행 2열 중에 2열은 막대그래프 출력
plot_value_func(i, pred, test_labels)

plt.show()





