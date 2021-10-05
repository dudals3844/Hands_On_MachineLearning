import tensorflow as tf


def cube(x):
    return x ** 3


print(cube(2))
print(cube(tf.constant(2)))

tf_cube = tf.function(cube)
print(tf_cube)

print(tf_cube(2))


@tf.function
def tf_cube(x):
    return x ** 3


print(tf_cube.python_function(2))

