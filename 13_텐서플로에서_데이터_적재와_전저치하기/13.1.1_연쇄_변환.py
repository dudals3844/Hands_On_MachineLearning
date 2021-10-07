import tensorflow as tf

X = tf.range(10)

dataset = tf.data.Dataset.from_tensor_slices(X)

dataset_repeat = dataset.repeat(3)
for item in dataset_repeat:
    # print(item)
    pass
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    # print(item)
    pass

dataset = tf.data.Dataset.range(10).repeat(3)
# buffer_size가 충분히 커야지 더 램덤한 값이 나온다
dataset = dataset.shuffle(buffer_size=100, seed=42).batch(5)
for item in dataset:
    # print(item)
    pass

dataset = tf.data.Dataset.range(10)
# dataset = dataset.map(lambda x: x * 2)
# for item in dataset:
#     print(item)

dataset = dataset.repeat(3).batch(7)

dataset = dataset.unbatch()
dataset = dataset.filter(lambda x: x < 10)

for item in dataset.take(3):
    print(item)