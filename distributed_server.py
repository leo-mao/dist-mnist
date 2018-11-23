# coding : utf-8
import tensorflow as tf
import time
import mnist_inference
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'Directory for mnist data')
flags.DEFINE_integer('hidden_units', 100, '')
flags.DEFINE_integer('training_steps', 4000, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_float('learning_rate_base', 1e-04, '')
flags.DEFINE_float('learning_rate_decay', 0.99, '')
flags.DEFINE_float('moving_average_decay', 0.99, '')
flags.DEFINE_float('regularaztion_rate', 1e-4, '')
flags.DEFINE_bool('is_sync', False, '')
# flags.DEFINE_string('data_path', True, '')
flags.DEFINE_string('model_save_path', './checkpoints', '')

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
    regularizer = tf.contrib.layers.l2_regularizer(FLAGS.regularaztion_rate)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.train.get_or_create_global_step()

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate_base,
        global_step,
        60000 / FLAGS.batch_size,  # 60000???TODO mnist.train.num_examples, every batch decays some extent
        FLAGS.learning_rate_decay)

    # tf.train.SyncReplicasOptimizer
    opt = tf.train.SyncReplicasOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate),
        replicas_to_aggregate=n_workers,
        total_num_replicas=n_workers
    )
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    train_op = opt.minimize(loss, global_step=global_step)

    if is_chief:
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op()

    return global_step, loss, train_op, sync_replicas_hook


def main(argv=None):
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

    device_setter = tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % FLAGS.task_index, cluster=cluster)

    with tf.device(device_setter):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        global_step, loss, train_op, sync_replicas_hook = build_model(x, y_, n_workers, is_chief)

        hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=FLAGS.training_steps)]
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    target_collection = []  # target tensor
    if is_chief:
        collection = tf.local_variables() + target_collection
    else:
        collection = tf.local_variables()
    local_init_op = tf.variables_initializer(collection)
    ready_for_local_init_op = tf.report_uninitialized_variables(collection)

    scaffold = tf.train.Scaffold(local_init_op=local_init_op, ready_for_local_init_op=ready_for_local_init_op)
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=FLAGS.model_save_path,
                                           hooks=hooks,
                                           save_checkpoint_secs=60,
                                           config=sess_config,
                                           scaffold=scaffold) as mon_sess:
        print('session started')
        step = 0
        start_time = time.time()

        while not mon_sess.should_stop():
            xs, ys = mnist.train.next_batch(FLAGS.batch_size)
            _, loss_value, global_step_value = mon_sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / global_step_value
                format_str = "After %d training steps (%d global steps), " + \
                             "loss on training batch is %g. (%.3f sec/batch)"
                print(format_str % (step, global_step_value, loss_value, sec_per_batch))
            step += 1


# def _DistributedInitializerHook(session_run_hook.SessionRunHook, variables_collection):
#     def __init__(self, initializer, variables_collection):
#         self._initializer = initializer
#         self._variables_collection = variables_collection
#     def begin(self):
#         pass


if __name__ == "__main__":
    tf.app.run()
