# coding : utf-8

import tensorflow as tf
import time
import mnist_inference
import tempfile
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'Directory for mnist data')
flags.DEFINE_integer('hidden_units', 100, '')
flags.DEFINE_integer('training_steps', 30000, '')
flags.DEFINE_integer('batch_size', 50, '')
flags.DEFINE_float('learning_rate_base', 1e-04, '')
flags.DEFINE_float('learning_rate_decay', 0.99, '')
flags.DEFINE_float('moving_average_decay', 0.99, '')
flags.DEFINE_float('regularaztion_rate', 1e-4, '')
flags.DEFINE_bool('is_sync', False, '')
# flags.DEFINE_string('data_path', True, '')
flags.DEFINE_string('model_save_path', None, '')

# workers' port numbers start from 9900
# while ps's port numbers start from 9910
# worker_0 referred as /job:worker/task:0
flags.DEFINE_string('ps_hosts', None, '')
flags.DEFINE_string('worker_hosts', None, '')
flags.DEFINE_string('job_name', None, 'worker or ps')
flags.DEFINE_integer('task_index', None, '')

FLAGS = flags.FLAGS
IMAGE_PIXELS = 28


def build_model(x, y_, n_workers, is_chief):
    with tf.name_scope('regularizer'):
        regularizer = tf.contrib.layers.l2_regularizer(FLAGS.regularaztion_rate)
    with tf.name_scope('y'):
        y = mnist_inference.inference(x, regularizer)
        tf.summary.histogram('y', y)
    with tf.name_scope('get_global_step'):
        global_step = tf.train.get_or_create_global_step()
    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate_base,
            global_step,
            FLAGS.training_steps,  # 60000???TODO mnist.train.num_examples, every batch decays some extent
            FLAGS.learning_rate_decay)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('cross entropy', loss)

    with tf.name_scope('back_propagation'):
        # tf.train.SyncReplicasOptimizer
        opt = tf.train.SyncReplicasOptimizer(
            tf.train.GradientDescentOptimizer(learning_rate),
            replicas_to_aggregate=n_workers,
            total_num_replicas=n_workers
        )
    with tf.name_scope('optimizer'):
        train_op = opt.minimize(loss, global_step=global_step)
    with tf.name_scope('accuracy_evaluation'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('sync_replicas_hook'):
        sync_replicas_hook = opt.make_session_run_hook(is_chief)

        if is_chief:
            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            with tf.control_dependencies([variables_averages_op, train_op]):
                train_op = tf.no_op()

        tf.summary.scalar('model/accuracy', accuracy)
        tf.summary.histogram('learning_rate', learning_rate)
    merge = tf.summary.merge_all()

    return merge, global_step, loss, train_op, sync_replicas_hook, accuracy


def main(argv):

    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    n_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()
    is_chief = (FLAGS.task_index == 0)
    mnist = read_data_sets('MNIST_data', one_hot=True)
    train_dir = FLAGS.model_save_path if FLAGS.model_save_path is not None else tempfile.mkdtemp()
    device_setter = tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % FLAGS.task_index, cluster=cluster)

    with tf.device(device_setter):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        summary, global_step, loss, train_op, sync_replicas_hook, accuracy = build_model(x, y_, n_workers, is_chief)

        with tf.name_scope('stop_hook'):
            hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=FLAGS.training_steps)]
        with tf.name_scope('Session_config'):
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    # input output
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=train_dir,
                                           hooks=hooks,
                                           save_checkpoint_secs=60,
                                           config=sess_config) as mon_sess:
        print('session started')
        step = 0
        start_time = time.time()
        train_write = tf.summary.FileWriter('./graphs/{}/'.format(FLAGS.task_index), tf.get_default_graph())

        while not mon_sess.should_stop():
            xs, ys = mnist.train.next_batch(FLAGS.batch_size)
            summary_value, _, loss_value, global_step_value, accuracy_value = mon_sess.run(
                [summary, train_op, loss, global_step, accuracy], feed_dict={x: xs, y_: ys})
            train_write.add_summary(summary_value, global_step_value)

            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / global_step_value
                format_str = "After %d training steps (%d global steps), " + \
                             "loss on training batch is %g. (%.3f sec/batch)" + \
                             "\n Accuracy on test set is (%.4f )"
                print(format_str % (step, global_step_value, loss_value, sec_per_batch, accuracy_value))

            step += 1
    train_write.close()


if __name__ == "__main__":
    tf.app.run()