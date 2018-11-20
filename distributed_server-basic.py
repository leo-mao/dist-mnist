import tensorflow as tf
import math
import tempfile
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'Directory for mnist data')
flags.DEFINE_integer('hidden_units', 100, '')
flags.DEFINE_integer('train_steps', 10000, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_float('learning_rate', 0.0001, '')


# workers' port numbers start from 9900
# while ps's port numbers start from 9910
# worker_0 referred as /job:worker/task:0
flags.DEFINE_string('ps_hosts', None, '')
flags.DEFINE_string('worker_hosts', None, '')
flags.DEFINE_string('job_name', None, 'worker or ps')
flags.DEFINE_integer('task_index', None, '')

FLAGS = flags.FLAGS
IMAGE_PIXELS = 28


def model_from_zhihu(images, labels):
    """The model of NN from https://zhuanlan.zhihu.com/p/35083779"""
    nn = tf.layers.dense(images, 500, activation=tf.nn.relu)
    nn = tf.layers.dense(nn, 500, activation=tf.nn.relu)
    nn = tf.layers.dense(nn, 10, activation=None)
    nn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn, labels=labels))
    return nn


def model_from_book_example(input, labels):

    hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                            stddev=1.0 / IMAGE_PIXELS), name='hid_w')
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

    sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                           stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
    sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

    activation = tf.nn.xw_plus_b(input, hid_w, hid_b)
    hid = tf.nn.relu(activation)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_mean(labels * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    return cross_entropy


def main(unused_argv):

    if FLAGS.job_name is not None and len(FLAGS.job_name) > 0:
        print('job name : {}'.format(FLAGS.job_name))
    else:
        raise ValueError('Must specify the job name explicitly')

    if FLAGS.task_index is not None and FLAGS.task_index >= 0:
        print('task index : {}'.format(FLAGS.task_index))
    else:
        raise ValueError('Must specify a valid task index')

    mnist = read_data_sets('MNIST_data', one_hot=True)

    worker_hosts = FLAGS.worker_hosts.split(',')
    num_workers = len(worker_hosts)
    ps_hosts = FLAGS.ps_hosts.split(',')

    cluster = tf.train.ClusterSpec({
        "worker": worker_hosts,
        "ps": ps_hosts
    })

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()

    if FLAGS.job_name == 'worker':

        # Between-graph replication
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:{}'.format(FLAGS.task_index),
                                                      cluster=cluster)):

            global_step = tf.Variable(0, name='global_step', trainable=False)
            input = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
            labels = tf.placeholder(tf.float32, [None, 10])

            # Model from Zhihu
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

            # Model from book
            loss = model_from_book_example(input, labels)

            hooks =[tf.train.StopAtStepHook(last_step=4000)]
            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_opt = opt.minimize(loss, global_step=global_step, aggregation_method=tf.AggregationMethod.ADD_N)


            train_dir = tempfile.mkdtemp()

            with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir=train_dir, hooks=hooks) as sess:
                while not sess.should_stop():
                    input_batch, label_batch = mnist.train.next_batch(32)
                    _, ls, step = sess.run([train_opt, loss, global_step],
                                           feed_dict={input: input_batch, labels: label_batch})

                    if step % 100 == 0:
                        print("Train step {}, loss: {}".format(step, ls))


if __name__ == '__main__':
    tf.app.run()