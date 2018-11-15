import tensorflow as tf

with tf.device('/cpu:0'):
    w=tf.get_variable('w', (2,2), tf.float32, initializer=tf.constant_initializer(2))
    b=tf.get_variable('b', (2,2), tf.float32, initializer=tf.constant_initializer(5))


with tf.device('/gpu:0'):

    addwb=w+b


with tf.device('/gpu:1'):

    mutwb=w*b
    init=tf.initialize_all_variables()


with tf.Session() as sess:

    sess.run(init)
    print(sess.run([addwb, mutwb]))