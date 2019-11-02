# coding:utf-8

import hashlib
import os.path
import re

import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.python.framework import tensor_shape

from constant import MAX_NUM_IMAGES_PER_CLASS

FLAGS = None


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """
    构建图片训练集
    Args：
        图片路径
        测试集百分比
        校验集百分比
    Returns：
        一个目录包含一个每个标签文件夹的入口，并根据标签分割成训练，测试，校验集
    """

    if not gfile.Exists(image_dir):
        tf.logging.error("图片目录 '" + image_dir + "' 找不到.")
        return None

    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # 跳过根目录
    is_root_dir = True

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)

        if dir_name == image_dir:
            continue

        tf.logging.info("在 '" + dir_name + " 查找图片'")

        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))

        if not file_list:
            tf.logging.warning('No files found')
            continue

        if len(file_list) < 20:
            tf.logging.warning('WARNING: Folder has less than 20 images, which may cause issues.')

        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name,
                                            MAX_NUM_IMAGES_PER_CLASS))

        # 目录名是中文所以编码成utf-8，用作key值
        label_name = dir_name.decode('utf-8')

        # 训练集
        training_images = []

        # 测试集
        testing_images = []

        # 校验集
        validation_images = []

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)

            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()

            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))

            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }

    return result


def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness,
                          input_width, input_height, input_depth, input_mean, input_std):
    """
    创建用于应用指定扭曲的操作。
    在训练过程中，如果我们运行的图像通过简单的扭曲，如裁剪，缩放和翻转，可以帮助改进结果。
    这些反映我们期望在现实世界中的变化，
    因此可以帮助训练模型，以更有效地应对自然数据。
    在这里，我们采取的供应参数并构造一个操作网络以将它们应用到图像中。
    裁剪
    ~~~~~~~~
    裁剪是通过在完整的图像上一个随机的位置放置一个边界框。裁剪参数控制该框相对于输入图像的尺寸大小。
    如果它是零，那么该框以输入图像相同的大小作为输入不进行裁剪。
    如果值是50%，则裁剪框将是输入的宽度和高度的一半。在图中看起来像这样：
    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+
    缩放
    ~~~~~~~
    缩放是非常像裁剪，除了边界框总是在中心和它的大小在给定的范围内随机变化。
    例如，如果缩放比例百分比为零，则边界框与输入尺寸大小相同，没有缩放应用。
    如果它是50%，那么边界框将是宽度和高度的一半和全尺寸之间的随机范围。
    Args：
        flip_left_right：是否随机镜像水平的布尔值。
        random_crop：在裁切框设置总的边缘的整数百分比。
        random_scale：缩放变化多少的整数百分比。
        random_brightness：随机像素值的整数范围。
    Returns：
        JPEG输入层和扭曲结果的张量。
    """

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)

    resize_scale_value = tf.random_uniform(tensor_shape.scalar(), minval=1.0, maxval=resize_scale)

    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [input_height, input_width, input_depth])

    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image

    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)

    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')

    return jpeg_data, distort_result
