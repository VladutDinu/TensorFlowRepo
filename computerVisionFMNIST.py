import tensorflow as tf
import numpy as np
from tensorflow import keras

fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels), (test_images, test_labels)=fashion_mnist.load_data()

train_images=train_images.reshape(60000,28,28,1)
train_images=train_images/255.0

test_images=test_images.reshape(10000,28,28,1)
test_images=test_images/255.0

model=keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')
model.summary()
model.fit(train_images,train_labels, epochs=7)
model.evaluate(test_images,test_labels)