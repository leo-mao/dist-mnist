import tensorflow as tf

cluster = tf.train.ClusterSpec({
    "worker":[
        # workers' port numbers start from 9900
        # worker_0 referred as /job:worker/task:0
        "127.0.0.1:9900",
        # worker_1 referred as /job:worker/task:1
        "127.0.0.1:9901",
        # worker_2 referred as /job:worker/task:2
        "127.0.0.1:9902"
    ],
    "ps":[
        # ps has a port number 9910
        # ps referred as /job:ps/task:0
        "127.0.0.1:9910"
    ]
})

with tf.device('/job:ps/task:0'):
    w=tf

server_worker_0 = tf.train.Server(cluster, job_name='worker', task_index=0)
server_worker_1 = tf.train.Server(cluster, job_name='worker', task_index=1)
server_worker_2 = tf.train.Server(cluster, job_name='worker', task_index=2)

