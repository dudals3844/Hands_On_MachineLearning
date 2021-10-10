import tensorflow as tf

with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first world")
    f.write(b"And this is the second record")

filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)