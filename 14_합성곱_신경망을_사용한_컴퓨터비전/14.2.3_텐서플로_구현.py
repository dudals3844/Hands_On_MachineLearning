import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
from tensorflow import keras

china = load_sample_image("china.jpg")
china = china / 255
flower = load_sample_image("flower.jpg") / 255
imges = np.array([china, flower])
batch_size, height, width, channels = imges.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # 수직선
filters[3, :, :, 1] = 1  # 수평선

print(filters)
outputs = tf.nn.conv2d(imges, filters, strides=1, padding="SAME")

plt.imshow(outputs[0, :, :, 1], cmap='gray')
plt.show()

np.random.seed(42)
tf.random.set_seed(42)

conv = keras.layers.Conv2D(filters=2, kernel_size=7, strides=1, padding="SAME", activation='relu',
                           input_dim=outputs.shape)

conv_outputs = conv(imges)

print(f"Imga Shape: {imges.shape}")
print(conv_outputs.shape)

