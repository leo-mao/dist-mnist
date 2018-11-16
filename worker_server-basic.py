import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cluster = tf.train.ClusterSpec({
    "worker": [
        # workers' port numbers start from 9900
        # worker_0 referred as /job:worker/task:0
        "127.0.0.1:9900",
        # worker_1 referred as /job:worker/task:1
        "127.0.0.1:9901",

    ],
    "ps": [
        # ps has a port number 9910
        # ps referred as /job:ps/task:0
        "127.0.0.1:9910"
    ]
})

isps = False


if isps:

    # Define the parameter server(ps)
    server = tf.train.Server(cluster, job_name='ps', task_index=0)
    server.join()

else:

    server = tf.train.Server(cluster, job_name='worker', task_index=0)
    # server.join()

    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:0', cluster=cluster)):
        w = tf.get_variable('w', (2, 2), tf.float32, initializer=tf.constant_initializer(2))
        b = tf.get_variable('b', (2, 2), tf.float32, initializer=tf.constant_initializer(5))
        addwb = w+b
        mutwb = w*b
        divwb = w/b

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    sv = tf.train.Supervisor(init_op=init_op, summary_op=summary_op, saver=saver)
    with sv.managed_session(server.target) as sess:
        while True:
            print(sess.run([addwb, mutwb, divwb]))


