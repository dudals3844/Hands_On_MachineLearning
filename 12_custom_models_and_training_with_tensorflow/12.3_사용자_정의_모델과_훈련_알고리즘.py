import sys

assert sys.version_info >= (3, 5)

import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

np.random.seed(42)
tf.random.set_seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - .5
    return tf.where(is_small_error, squared_loss, linear_loss)


# plt.figure(figsize=(8, 3.5))
# z = np.linspace(-4, 4, 200)
# plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
# plt.plot(z, z ** 2 / 2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
# plt.plot([-1, -1], [0, huber_fn(0., -1.)], "r--")
# plt.plot([1, 1], [0, huber_fn(0., 1.)], "r--")
# plt.gca().axhline(y=0, color='k')
# plt.gca().axvline(x=0, color='k')
# plt.axis([-4, 4, 0, 4])
# plt.grid(True)
# plt.xlabel("$z$")
# plt.legend(fontsize=14)
# plt.title("Huber loss", fontsize=14)
# plt.show()

input_shape = X_train.shape[1:]


# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
#                        input_shape=input_shape),
#     keras.layers.Dense(1),
# ])
#
# model.compile(loss=huber_fn, optimizer='nadam', metrics=['mse'])
# model.fit(X_train_scaled, y_train, epochs=3, validation_data=(X_valid_scaled, y_valid))
# model.save("my_model_with_a_custom_loss.h5")
#
# # model = keras.models.load_model("my_model_with_a_custom_loss.h5",
# #                                 custom_objects={"huber_fn": huber_fn})
#
# model.fit(X_train_scaled, y_train, epochs=2,
#           validation_data=(X_valid_scaled, y_valid))


def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    return huber_fn


#
# model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])
# model.fit(X_train_scaled, y_train, epochs=2,
#           validation_data=(X_valid_scaled, y_valid))
#
# model.fit(X_train_scaled, y_train, epochs=2,
#           validation_data=(X_valid_scaled, y_valid))


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold ** 2 / 2
        # is_small_error가 True이면 squared_loss 출력 아니면 linear_loss 출k
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


#
# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
#                        input_shape=input_shape),
#     keras.layers.Dense(1),
# ])
#
# print('Class HuberLoss')
# model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])
# model.fit(X_train_scaled, y_train, epochs=2,
#           validation_data=(X_valid_scaled, y_valid))
# # model = keras.models.load_model("my_model_with_a_custom_loss_class.h5", custom_objects={'HuberLoss':HuberLoss})
# print(model.loss.threshold)
#
#
def my_softplus(z):
    return tf.math.log(tf.exp(z) + 1.0)


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))


def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


#
# layer = keras.layers.Dense(30, activation=my_softplus,
#                            kernel_initializer=my_glorot_initializer,
#                            kernel_regularizer=my_l1_regularizer,
#                            kernel_constraint=my_positive_weights)
#
# keras.backend.clear_session()
# np.random.seed(42)
# tf.random.set_seed(42)
#
# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation='selu', kernel_initializer='lecun_normal', input_shape=input_shape),
#     keras.layers.Dense(1, activation=my_softplus,
#                        kernel_regularizer=my_l1_regularizer,
#                        kernel_constraint=my_positive_weights,
#                        kernel_initializer=my_glorot_initializer),
# ])
#
# model.compile(loss='mse', optimizer='nadam', metrics=['mae'])
# model.fit(X_train_scaled, y_train, epochs=2,
#           validation_data=(X_valid_scaled, y_valid))
#
#
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}


#
# keras.backend.clear_session()
#
# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation='relu', kernel_initializer='lecun_normal', input_shape=input_shape),
#     keras.layers.Dense(1, activation=my_softplus,
#                        kernel_regularizer=MyL1Regularizer(0.01),
#                        kernel_constraint=my_positive_weights,
#                        kernel_initializer=my_glorot_initializer
#                        )
# ])
#
# model.compile(loss='mse', optimizer='nadam', metrics=['mae'])
# model.fit(X_train_scaled, y_train, epochs=3, validation_data=(X_valid_scaled, y_valid))
#
# model.save('my_model.h5')
#
# # 사용자 정의 지표
# keras.backend.clear_session()
# np.random.seed(42)
# tf.random.set_seed(42)
#
# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation='selu', kernel_initializer='lecun_normal', input_shape=input_shape),
#     keras.layers.Dense(1)
# ])
#
# model.compile(loss='mse', optimizer='nadam', metrics=[create_huber(2.0)])
# model.fit(X_train_scaled, y_train, epochs=2)
#
# model.compile(loss=create_huber(2.0), optimizer='nadam', metrics=[create_huber(2.0)])
# sample_weights = np.random.rand(len(y_train))
# history = model.fit(X_train_scaled, y_train, epochs=2, sample_weight=sample_weights)
#
# history.history['loss'][0], history.history['huber_fn']

# 스트리밍 지표
precision = keras.metrics.Precision()
precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
print(precision.result())

precision([1, 1, 1, 1], [1, 1, 1, 0])
print(precision.result())
print(precision.variables)


class HuberMetric(keras.metrics.Mean):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


m = HuberMetric(2.)
m(tf.constant([[2.]]), tf.constant([[10.]]))
print(m.result())

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation='selu', kernel_initializer='lecun_normal', input_shape=input_shape),
#     keras.layers.Dense(1),
# ])
#
# model.compile(loss=keras.losses.Huber(2.), optimizer='nadam', weighted_metrics=[HuberMetric(2.)])
# sample_weight = np.random.rand(len(y_train))
#
# history = model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)
# print(model.metrics[-1].threshold)

# 사용자 정의 층
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
print(exponential_layer([-1., 0., 1.]))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation='relu', input_shape=input_shape),
#     keras.layers.Dense(1),
#     exponential_layer
# ])
#
# model.compile(loss='mse', optimizer='sgd')
# model.fit(X_train_scaled, y_train, epochs=5, validation_data=(X_valid_scaled, y_valid))
# model.evaluate(X_test_scaled, y_test)

class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name='kernel', shape=[batch_input_shape[-1], self.units],
            initializer='glorot_normal')
        self.bias = self.add_weight(
            name='bias', shape=[self.units], initializer='zeros')
        super().build(batch_input_shape)

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def get_output_shape_at(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}


# keras.activations.serialize 란
print(tf.keras.activations.serialize(tf.keras.activations.sigmoid))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    MyDense(30, activation='relu', input_shape=input_shape),
    MyDense(1)
])

model.compile(loss='mse', optimizer='nadam')
model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)

model.save("my_model_with_a_custom_layer.h5")

class MyMultiLayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        print(f"X1.shape: {X1.shape}, X2.shape: {X2.shape}")
        return X1 + X2, X1 * X2

    def compute_output_shape(self, batch_input_shape):
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape2]

inputs1 = keras.layers.Input(shape=[2])
inputs2 = keras.layers.Input(shape=[2])
outputs1, outputs2 = MyMultiLayer()((inputs1, inputs2))

def split_data(data):
    columns_count = data.shape[-1]
    half = columns_count // 2
    return data[:, :half], data[:, half:]

X_train_scaled_A, X_train_scaled_B = split_data(X_train_scaled)
X_valid_scaled_A, X_valid_scaled_B = split_data(X_valid_scaled)
X_test_scaled_A, X_test_scaled_B = split_data(X_test_scaled)

# 분할된 데이터 크기 출력
print(X_train_scaled_A.shape, X_train_scaled_B.shape)

outputs1, outputs2 = MyMultiLayer()((X_train_scaled_A, X_train_scaled_B))
print(outputs1, outputs2)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# input_A = keras.layers.Input(shape=X_train_scaled_A.shape[-1])
# input_B = keras.layers.Input(shape=X_train_scaled_B.shape[-1])
# hidden_A, hidden_B = MyMultiLayer()((input_A, input_B))
# hidden_A = keras.layers.Dense(30, activation='selu')(hidden_A)
# hidden_B = keras.layers.Dense(30, activation='selu')(hidden_B)
# concat = keras.layers.Concatenate()((hidden_A, hidden_B))
# output = keras.layers.Dense(1)(concat)
# model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
#
# model.compile(loss='mse', optimizer='nadam')
# model.fit((X_train_scaled_A, X_train_scaled_B), y_train, epochs=2, validation_data=((X_valid_scaled_A, X_valid_scaled_B), y_valid))

class AddGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X
    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

model =