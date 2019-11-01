# coding:utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
# train_dir = os.environ['IMAGE_TRAIN_INPUT_PATH']
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784], name="Input")  # 为输入op添加命名"Input"
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))
tf.identity(y, name="Output")  # 为输出op命名为"Output"

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

# 将模型保存到文件
# 简单方法：
MODEL_PATH = os.environ['MODEL_INFERENCE_PATH']
tf.saved_model.simple_save(
    tf.keras.backend.get_session(),
    MODEL_PATH + "/SavedModel/",
    inputs={"Input": x},
    outputs={"Output": y}
    # inputs={"your_model_inputs": model.input},
    # outputs={"your_model_outputs": model.output}
)
