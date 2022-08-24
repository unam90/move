# zoo dataset으로 동물의 type 분류 
# 1. animal name: Unique for each instance
# 2. hair: Boolean
# 3. feathers: Boolean
# 4. eggs: Boolean
# 5. milk: Boolean
# 6. airborne: Boolean
# 7. aquatic: Boolean
# 8. predator: Boolean
# 9. toothed: Boolean
# 10. backbone: Boolean
# 11. breathes: Boolean
# 12. venomous: Boolean
# 13. fins: Boolean
# 14. legs: Numeric (set of values: {0,2,4,5,6,8})
# 15. tail: Boolean
# 16. domestic: Boolean
# 17. catsize: Boolean
# 18. type: Numeric (integer values in range [1,7])

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

xy = np.loadtxt('../testdata/zoo.csv', delimiter=',')
print(xy[0], xy.shape)  # (101, 17)

x_data = xy[:, 0:-1]  # feature
y_data = xy[:, -1]    # label
print(x_data[:3])
print(y_data[:3])
print(set(y_data))  # {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

nb_classes = 7 
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print(y_one_hot[:3])

# model
model = Sequential()
model.add(Dense(32, input_shape=(16, ), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(x_data, y_one_hot, epochs=100, batch_size=10, 
                    validation_split=0.3, verbose=0)

print('eval:', model.evaluate(x_data, y_one_hot))

# 시각화
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

import matplotlib.pyplot as plt
plt.plot(loss, 'b-', label='train_loss')
plt.plot(val_loss, 'r--', label='val_loss')
plt.legend()
plt.show()

plt.plot(acc, 'b-', label='train_acc')
plt.plot(val_acc, 'r--', label='val_acc')
plt.legend()
plt.show()

# predict
pred_data = x_data[:1]
pred = np.argmax(model.predict(pred_data))
print('pred:', pred)

print()
pred_datas = x_data[:5]
preds = [np.argmax(i) for i in model.predict(pred_datas)]
print('예측값들:', preds)
print('실제값들:', y_data[:5])
