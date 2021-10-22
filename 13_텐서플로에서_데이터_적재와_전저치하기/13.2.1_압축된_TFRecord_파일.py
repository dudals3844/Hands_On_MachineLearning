import tensorflow as tf

options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"], compression_type="GZIP")

for item in dataset:
    print(item)

