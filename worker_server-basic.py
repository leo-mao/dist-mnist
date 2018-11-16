import tensorflow as tf
import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags

flags.DEFINE_string('data_dir', '', 'Directory for mnist data')
flags.DEFINE_integer('hidden_units', 100, '')
flags.DEFINE_integer('train_steps', 10000, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_float('learning_rate', 0.01, '')

flags.DEFINE_string('ps_hosts', '127.0.0.1:9910', '')
# workers' port numbers start from 9900
# worker_0 referred as /job:worker/task:0
flags.DEFINE_string('worker_hosts', '127.0.0.1:9900,127.0.0.1:9901', '')
flags.DEFINE_string('job_name', None, 'worker or ps')
flags.DEFINE_integer('task_index', None, '')

FLAGS = flags.FLAGS


def main(unused_argv):

    if FLAGS.job_name is not None and len(FLAGS.job_name) > 0:
        print('job name : {}'.format(FLAGS.job_name))
    else:
        raise ValueError('Must specify the job name explicitly')

    if FLAGS.task_index is not None and FLAGS.task_index >= 0:
        print('task index : {}'.format(FLAGS.task_index))
    else:
        raise ValueError('Must specify a valid task index')
    worker_hosts = FLAGS.worker_hosts.split(',')
    ps_hosts = FLAGS.ps_hosts.split(',')


    cluster = tf.train.ClusterSpec({
        "worker": worker_hosts,
        "ps": ps_hosts
    })

    num_worker = len(worker_hosts)
    server = tf.train.Server(cluster, job_name='ps', task_index=FLAGS.job_index)

    if FLAGS.job_name == 'ps':
        server.join()

    if FLAGS.job_name == 'worker':
        server = tf.train.Server(cluster, job_name='worker', task_index=FLAGS.job_index)
        is_chief = (FLAGS.job_index == 0)
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):

            global_step = tf.Variable(0, name='global_step', trainable=False)

            w = tf.get_variable('w', (2, 2), tf.float32, initializer=tf.constant_initializer(2))
            b = tf.get_variable('b', (2, 2), tf.float32, initializer=tf.constant_initializer(5))
            addwb = w+b
            mutwb = w*b
            divwb = w/b

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(init_op=init_op, summary_op=summary_op, saver=saver)

        if is_chief:
            print('Worker {}: Initializing session...'.format(FLAGS.task_index))
        else:
            print('Worker {}: Waiting for session to be initialized...'.format(FLAGS.task_index))
        with sv.managed_session(server.target) as sess:
            while True:
                print(sess.run([addwb, mutwb, divwb]))


if __name__ == '__main__':
    tf.app.run()