#!/usr/bin/env python3

import tensorflow.keras as kr
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def preprocess(image):
	image = image.resize((28, 28), Image.BICUBIC)
	pix = np.fromiter(image.getdata(), 'float32')
	pix = pix.reshape(1, 28, 28, 1)
	return 1 - pix/255.0

im = Image.open('img.png')
net_input = preprocess(im)

net = kr.models.load_model('mnist-model.h5')
pred = net.predict(net_input)[0]
digit = np.argmax(pred)
conf = pred[digit]

plt.figure()
plt.subplot(1, 2, 1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(net_input.reshape(28, 28), cmap=plt.cm.binary)
plt.xlabel('%d (%2.0f%%)' % (digit, conf*100))
plt.subplot(1, 2, 2)
plt.xticks(range(10))
plt.yticks([])
plt.grid(False)
plot = plt.bar(range(10), pred)
plt.ylim([0, 1])
plot[digit].set_color('blue')
plt.show()
