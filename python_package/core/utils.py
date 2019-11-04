# coding:utf-8

import os
import sys
import random
import tarfile
import urllib
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from constant import MAX_NUM_IMAGES_PER_CLASS

FLAGS = None


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"
    返回给定索引中的标签的瓶颈文件的路径。
    Args：
        image_lists：训练图像每个标签的词典。
        label_name：我们想得到的一个图像的标签字符串。
        index：我们想要图像的Int 偏移量。这将以标签的可用的图像数为模，因此它可以任意大。
        bottleneck_dir：文件夹字符串保持缓存文件的瓶颈值。
        category：从图像训练、测试或验证集提取的图像的字符串名称。
    Returns：
        将文件系统路径字符串映射到符合要求参数的图像。
    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_random_distorted_bottlenecks(sess, image_lists, how_many, category,
                                     image_dir, input_jpeg_tensor,
                                     distorted_image, resized_input_tensor,
                                     bottleneck_tensor):
    """
    检索训练图像扭曲后的瓶颈值。
    如果我们训练使用扭曲变换，如裁剪，缩放，或翻转，我们必须重新计算每个图像的完整模型，
    所以我们不能使用缓存的瓶颈值。相反，我们找出所要求类别的随机图像，
    通过扭曲图运行它们，然后得到每个瓶颈结果完整的图。
    Args：
        sess：当前的tensorflow会话。
        image_lists：每个标签的训练图像的词典。
        how_many：返回瓶颈值的整数个数。
        category：要获取的图像训练、测试，或验证集的名称字符串。
        image_dir：包含训练图像的子文件夹的根文件夹字符串.
        input_jpeg_tensor：给定图像数据的输入层。
        distorted_image：畸变图形的输出节点。
        resized_input_tensor：识别图的输入节点。
        bottleneck_tensor：CNN图的瓶颈输出层。
    Returns:
        瓶颈阵列及其对应的ground truths列表。
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        # Note that we materialize the distorted_image_data as a numpy array before
        # sending running inference on the image. This involves 2 memory copies and
        # might be optimized in other implementations.
        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(
            bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck_values)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def download_and_extract(data_url):
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
