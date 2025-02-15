import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

#test
''' 
if X_train.shape == (60000, 28, 28) and Y_train.shape == (60000,):
    print("test ok")

for i in range(12):  
  plt.subplot(4,3,i+1)
  plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()
'''

X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

model = tf.keras.models.Sequential()
Conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu')
model.add(Conv1)
MaxPool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
model.add(MaxPool1)
Conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu')
model.add(Conv2)
MaxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
model.add(MaxPool2)
model.add(tf.keras.layers.Flatten())
Dense1 = tf.keras.layers.Dense(units = 128, activation='relu')
model.add(Dense1)
Dense2 = tf.keras.layers.Dense(units = 10, activation='softmax')
model.add(Dense2)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs = 3)

model.save('Handwritten_Digit_AI.model')