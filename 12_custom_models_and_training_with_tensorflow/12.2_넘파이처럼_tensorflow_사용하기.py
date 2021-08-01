import sys

assert sys.version_info >= (3, 5)

import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np

import os

np.random.seed(42)
tf.random.set_seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deep"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
tf.constant(42)

# print(t.shape)
# print(t.dtype)

# 인덱싱
# print(t[:, 1:])

# 연산
# print(t + 10)

# print(tf.square(t))

# print(t @ tf.transpose(t))

from tensorflow import keras

K = keras.backend
# print(t)
# print(tf.transpose(t))
# print(K.square(K.transpose(t)) + 10)

# Numpy 변환
a = np.array([2., 4., 5.])
# print(a)
# print(tf.constant(a))

# print(t.numpy())

# Type Casting
# try:
#     tf.constant(2.0) + tf.constant(40)
# except tf.errors.InvalidArgumentError as ex:
#     print(ex)
#
# try:
#     tf.constant(2.0) + tf.constant(40., dtype=tf.float64)
# except tf.errors.InvalidArgumentError as ex:
#     print(ex)

t2 = tf.constant(40, dtype=tf.float64)
tmp = tf.constant(2.0) + tf.cast(t2, tf.float32)
# print(tmp)

s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]], values=[1., 2., 3.], dense_shape=[3, 4])
# print(s)
# print(tf.sparse.to_dense(s))

s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
result = tf.sparse.sparse_dense_matmul(s, s4)
# print(s)
# print(tf.transpose(s4))
# print(result)

# print(tf.sparse.to_dense(s) @ s4)
s5 = tf.SparseTensor(indices=[[0, 2], [0, 1]],
                     values=[1., 2.],
                     dense_shape=[3, 4])
# print(s5)

s6 = tf.sparse.reorder(s5)
ss = tf.sparse.to_dense(s6)
# print(ss)

# 집합
set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
set2 = tf.constant([[4, 5, 6], [9, 10, 0]])

union = tf.sparse.to_dense(tf.sets.union(set1, set2))
# print(union)

diff = tf.sparse.to_dense(tf.sets.difference(set1, set2))
# print(diff)

intersact = tf.sparse.to_dense(tf.sets.intersection(set1, set2))
print(intersact)

# 변수
v = tf.Variable([[1., 2., 3.], [4., 5, 6.]])
# print(v.assign(2 * v)

# print(v[0, 1].assign(42))
# print(v)

# [0,0]을 100으로 [1,2]를 200으로 update
# print(v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.]))

sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],
                                indices=[1, 0])
# print(v.scatter_update(sparse_delta))

# Tensor 배열
array = tf.TensorArray(dtype=tf.float32, size=3)
array = array.write(0, tf.constant([1., 2.]))
array = array.write(1, tf.constant([3., 10.]))
array = array.write(2, tf.constant([5., 7.]))

# print(array.read(1))

mean, variance = tf.nn.moments(array.stack(), axes=0)
print(mean)
print(variance)

