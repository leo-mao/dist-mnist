import tensorflow as tf

cluster = tf.train.ClusterSpec({
    "worker": [
        # workers' port numbers start from 9900
        # worker_0 referred as /job:worker/task:0
        "127.0.0.1:9900",
        # worker_1 referred as /job:worker/task:1
        "127.0.0.1:9901",
        # worker_2 referred as /job:worker/task:2
        "127.0.0.1:9902"
    ],
    "ps": [
        # ps has a port number 9910
        # ps referred as /job:ps/task:0
        "127.0.0.1:9910"
    ]
})


# Define the parameter server(ps)
server_ps = tf.train.Server(cluster, job_name='ps', task_index=0)
server_ps.join()

# Define the tasks(servers)
# server_worker_0 now has a reference 'task:0' in worker
server_worker_0 = tf.train.Server(cluster, job_name='worker', task_index=0)
# server_worker_1 now has a reference 'task:1' in worker
server_worker_1 = tf.train.Server(cluster, job_name='worker', task_index=1)
# server_worker_2 now has a reference 'task:2' in worker
server_worker_2 = tf.train.Server(cluster, job_name='worker', task_index=2)

server_worker_0.join()
server_worker_1.join()
server_worker_2.join()


# Set parameters on ps
with tf.device('/job:ps/task:0'):
    w = tf.get_variable('w', (2,2), tf.float32, initializer=tf.constant_initializer(2))
    b = tf.get_variable('b', (2,2), tf.float32, initializer=tf.constant_initializer(5))

# Assign operations to devices on tasks
with tf.device('/job:worker/task:0/gpu:0'):
    addwb = w+b

with tf.device('/job:worker/task:1/gpu:0'):
    mutwb = w*b

with tf.device('/job:worker/task:2/gpu:0'):
    divwb = w/b

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run([addwb, mutwb, divwb]))

