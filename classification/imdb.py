#!/usr/bin/env python3

import tensorflow.keras as kr
import numpy as np
import matplotlib.pyplot as plt

vocab_size = 10000
imdb = kr.datasets.imdb
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=vocab_size)
words = imdb.get_word_index()
words = {k: (v + 3) for k, v in words.items()}
words['<PAD>'] = 0
words['<START>'] = 1
words['<UNKNOWN>'] = 2
words['<UNUSED>'] = 3
reverse_words = {v: k for k, v in words.items()}

def decode(text):
	lookup = lambda x: reverse_words.get(x, '???')
	words = map(lookup, text)
	return ' '.join(words)

def preprocess(text):
	return kr.preprocessing.sequence.pad_sequences(text, value=0,
		padding='post', maxlen=256)

train_x = preprocess(train_x)
test_x = preprocess(test_x)

net = kr.Sequential([
	kr.layers.Embedding(vocab_size, 16),
	kr.layers.GlobalAveragePooling1D(),
	kr.layers.Dense(16, kernel_regularizer=kr.regularizers.l2(0.001),
		activation='relu'),
	kr.layers.Dropout(0.3),
	kr.layers.Dense(16, kernel_regularizer=kr.regularizers.l2(0.001),
		activation='relu'),
	kr.layers.Dropout(0.3),
	kr.layers.Dense(1, activation='sigmoid'),
])
net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = net.fit(train_x, train_y, epochs=12, batch_size=512,
	validation_data=(test_x, test_y))

accuracy = history.history['acc']
validation_accuracy = history.history['val_acc']
epochs = range(1, len(accuracy) + 1)

plt.figure()
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, '--', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


