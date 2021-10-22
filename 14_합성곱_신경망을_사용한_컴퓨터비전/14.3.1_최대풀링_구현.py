import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
from tensorflow import keras

china = load_sample_image("china.jpg")
china = china / 255

def crop(images):
    return images[150:220, 130:250]

max_pool = keras.layers.MaxPool2D(pool_size=2)

cropped_images = np.array([crop(image) for image in [china]], dtype=np.float32)
output = max_pool(cropped_images)

fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Input", fontsize=14)
ax1.imshow(cropped_images[0])  # 첫 번째 이미지 그리기
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Output", fontsize=14)
ax2.imshow(output[0])  # 첫 번째 이미지 출력 그리기
ax2.axis("off")
plt.savefig('tmp.jpg')
plt.show()