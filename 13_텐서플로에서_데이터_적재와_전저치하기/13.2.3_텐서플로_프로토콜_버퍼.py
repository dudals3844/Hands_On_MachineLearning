import tensorflow as tf

person_exmaple = tf.train.Example(
    features = tf.train.Features(
        feature={
            "name": tf.train.Feature(bytes_list = tf.train.BytesList(value=[b"Alice"])),
            "id": tf.train.Feature(int64_list = tf.train.Int64List(value=[123])),
            "emails": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"a@b.com", b"c@d.com"]))
        }
    )
)

with tf.io.TFRecordWriter('my_contacts.tfrecord') as f:
    f.write(person_exmaple.SerializeToString())