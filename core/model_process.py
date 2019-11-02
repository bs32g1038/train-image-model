# coding:utf-8
import os
import tensorflow as tf
from constant import LEARNING_RATE, MODEL_DIR, FINAL_TENSOR_NAME
from config import MODEL_INFO
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util


def create_model_graph():
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(MODEL_DIR, MODEL_INFO['model_file_name'])
        print(model_path, "asssss")
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    MODEL_INFO['bottleneck_tensor_name'],
                    MODEL_INFO['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


def variable_summaries(var):
    """附加一个张量的很多总结（为tensorboard可视化）。"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
    """
    为训练增加了一个新的softmax和全连接层。
    我们需要重新训练顶层识别我们新的类，所以这个函数向图表添加正确的操作，以及一些变量来保持
    权重，然后设置所有的梯度向后传递。
    softmax和全连接层的设置是基于：
        https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    Args：
        class_count：我们需要识别多少种类东西的整数数目。
        final_tensor_name：产生结果时新的最后节点的字符串名称。
        bottleneck_tensor：主CNN图像的输出。
    Returns：
        训练的张量和交叉熵的结果，瓶颈输入和groud truth输入的张量。
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor,
                                                       shape=[None, bottleneck_tensor_size],
                                                       name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

    # 组织以下的ops作为‘final_training_ops’,这样在TensorBoard里更容易看到。
    layer_name = 'final_training_ops'

    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')

            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]),
                                       name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input,
            ground_truth_input, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """
    插入我们需要的操作，以评估我们结果的准确性。
    Args：
      result_tensor：产生结果的新的最后节点。
      ground_truth_tensor：我们提供的groud truth数据的节点。
    Returns：
      元组（评价步骤，预测）。
     """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return
