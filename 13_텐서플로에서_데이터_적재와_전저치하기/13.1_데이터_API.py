import tensorflow as tf

X = tf.range(10)

dataset = tf.data.Dataset.from_tensor_slices(X)
print(dataset)

for item in dataset:
    print(item)