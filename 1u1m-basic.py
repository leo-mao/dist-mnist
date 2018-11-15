import tensorflow as tf
from tensorflow.python.client import device_lib
# from tensorflow.python.platform.test import gpu_device_name


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())

with tf.device('/cpu:0'):
    w = tf.get_variable('w', (2, 2), tf.float32, initializer=tf.constant_initializer(2))
    b = tf.get_variable('b', (2, 2), tf.float32, initializer=tf.constant_initializer(5))

with tf.device('/gpu:0'):
    addwb = w+b
    mutwb = w*b

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    np1, np2 = sess.run([addwb, mutwb])
    print(np1)
    print(np2)
