import tensorflow as tf
import math
import tempfile
import sys
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.ops import variables
# from tensorflow.python.training.training_util import get_global_step
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'Directory for mnist data')
flags.DEFINE_integer('hidden_units', 100, '')
flags.DEFINE_integer('train_steps', 4000, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_float('learning_rate', 1e-04, '')
flags.DEFINE_bool('is_sync', False, '')

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
    nn = tf.layers.dense(images, 500, activation=tf.nn.relu, name='relu_1')
    nn = tf.layers.dense(nn, 500, activation=tf.nn.relu, name='relu_2')
    nn = tf.layers.dense(nn, 10, activation=None, name='predict')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn, labels=labels))
    return cross_entropy, nn


def model_from_book_example(inputs, labels):
    hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                            stddev=1.0 / IMAGE_PIXELS), name='hid_w')
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

    sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                           stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
    sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

    activation = tf.nn.xw_plus_b(inputs, hid_w, hid_b)
    hid = tf.nn.relu(activation)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_mean(labels * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    return cross_entropy, y


def model_98_csdn_example(input, labels, keep_prob):
    """Add dropout layer https://blog.csdn.net/wangsiji_buaa/article/details/80205629"""
    W1 = variables.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS], stddev=1.0))
    b1 = variables.Variable(tf.zeros([500]) + 0.1)
    L1 = tf.matmul(input, W1) + b1
    L1_drop = tf.nn.dropout(L1, keep_prob=keep_prob)

    W2 = variables.Variable(tf.truncated_normal([500, 300], stddev=0.1))
    b2 = variables.Variable(tf.Zeros([300] + 0.1))
    L2 = tf.matmul(input, L1_drop, W2) + b2
    L2_drop = tf.nn.dropout(L2, keep_prob=keep_prob)

    W3 = variables.Variable(tf.truncated_normal([300, 10], stddev=0.1))
    b3 = variables.Variable(tf.zeros([10]) + 0.1)
    prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=prediction))
    return cross_entropy, prediction


def main(unused_args):

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
    is_chief = (FLAGS.task_index == 0)
    cluster = tf.train.ClusterSpec({
        "worker": worker_hosts,
        "ps": ps_hosts
    })

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()
        sys.exit('0')
    if FLAGS.job_name == 'worker':

        # Between-graph replication
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:{}'.format(FLAGS.task_index),
                                                      cluster=cluster)):

            # 1.Define the model
            inputs = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
            labels = tf.placeholder(tf.float32, [None, 10])
            keep_prob = tf.placeholder(tf.float32)
            input_test = mnist.test.images
            labels_test = mnist.test.labels

            # Model from Zhihu
            loss, y_predict = model_from_zhihu(images=inputs, labels=labels)
            # global_step = get_global_step()

            # Model from book
            # loss, y_predict = model_from_book_example(input, labels)
            global_step = tf.contrib.framework.get_or_create_global_step()


            # 2. Define the hook for initialization and queues
            hooks = [tf.train.StopAtStepHook(last_step=FLAGS.train_steps)]

            # 3. Train and test accuracy
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 50, 0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(labels, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            train_dir = tempfile.mkdtemp()

            # sync mode is currently unavailable, due to absence of the function get_or_create_global_step/
            #  get_global step
            # if FLAGS.is_sync:
            #     optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=num_workers,
            #                                                total_num_replicas=num_workers, name="sync_replicas")
            #     if is_chief:
            #         hooks.append(optimizer.make_session_run_hook(is_chief))

            train_opt = optimizer.minimize(loss, global_step=global_step, aggregation_method=tf.AggregationMethod.ADD_N)

            write = tf.summary.FileWriter('./graphs/dist-mnist.summary', tf.get_default_graph())

            with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,
                                                   checkpoint_dir=train_dir, hooks=hooks) as sess:
                while not sess.should_stop():
                    input_batch, label_batch = mnist.train.next_batch(FLAGS.batch_size)
                    _, ls, step = sess.run([train_opt, loss, global_step],
                                           feed_dict={inputs: input_batch, labels: label_batch})

                    if step % 100 == 0:
                        print('Train step {}, loss: {}'.format(step, ls))
                        if step < FLAGS.train_steps:
                            print('Accuracy: {}'.format(accuracy_op.eval({inputs: input_test,
                                                                          labels: labels_test}, session=sess)))
                write.close()


if __name__ == '__main__':
    tf.app.run()
