import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


# def create_nn_layer(input_tensor, input_dim, output_dim, layer_name, regularizer, act=tf.nn.relu):
#     with tf.name_scope(layer_name):
#         with tf.name_scope(layer_name):
#             weights = get_weight_variable(regularizer, input_dim, output_dim)
#             variable_summaries(weights, layer_name + '/weights')
#         with tf.name_scope('biases'):
#             biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
#             variable_summaries(biases, layer_name + '/biases')
#         with tf.name_scope('Wx_plus_b'):
#             preactivate = tf.matmul(input_tensor, weights) + biases
#             tf.summary.histogram(layer_name + '/pre_activations', preactivate)
#
#         activations = act(preactivate, name='activation')
#         tf.summary.histogram(layer_name + '/activation', activations)
#         return activations


def get_weight_variable(regularizer, input_dim, output_dim):
    with tf.name_scope('initialize_weights'):
        weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer, weights1, biases1, weights2, biases2):
    if regularizer is None:

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, regularizer(weights1)) + biases1)
        return tf.matmul(layer1, regularizer(weights2)) + biases2
