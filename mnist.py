#!/usr/bin/env python3

import tensorflow.keras as kr
import numpy as np
import matplotlib.pyplot as plt

def preprocess(images):
	images = images.reshape(images.shape[0], 28, 28, 1)
	return images / 255.0

classes = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
    'Shirt', 'Sneaker', 'Bag', 'Ankle Boot' ]

dataset = kr.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
train_images = preprocess(train_images)
test_images = preprocess(test_images)

model = kr.Sequential([
	kr.layers.Conv2D(16, 5, activation='relu', input_shape=(28, 28, 1)),
	kr.layers.Conv2D(32, 5, activation='relu'),
	kr.layers.MaxPooling2D(pool_size=(2, 2)),
	kr.layers.Flatten(),
	kr.layers.Dense(128, activation='relu'),
	kr.layers.Dense(128, activation='relu'),
	kr.layers.Dropout(0.2),
	kr.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5,
	validation_data=(test_images, test_labels))

accuracy = history.history['acc']
validation_accuracy = history.history['val_acc']
epochs = range(1, len(accuracy) + 1)

plt.figure()
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, 'g', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
