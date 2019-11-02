# coding:utf-8

import tensorflow as tf
from constant import SUMMARIES_DIR, IMAGE_DIR, TESTING_PERCENTAGE, VALIDATION_PERCENTAGE, FINAL_TENSOR_NAME
from core.model_process import create_model_graph, add_final_training_ops, save_graph_to_file
from core.image_process import create_image_lists, add_input_distortions
from config import MODEL_INFO
from utils import download_and_extract

# 图片扭曲参数，即是几何变换
IMAGE_PROCESS_VALUE = {
    'flip_left_right': False,
    'random_crop': 0,
    'random_scale': 0,
    'random_brightness': 0
}


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(SUMMARIES_DIR)
    tf.gfile.MakeDirs(SUMMARIES_DIR)
    return


def init():
    # 设置输出日志信息级别
    tf.logging.set_verbosity(tf.logging.INFO)
    prepare_file_system()

    # 获取图片列表
    image_lists = create_image_lists(IMAGE_DIR, TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)

    # 检测文件夹是否符合训练的规则
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('找不到该文件目录：' + IMAGE_DIR)
        return -1
    if class_count == 1:
        tf.logging.error('只有一个有效的图片文件夹 ' + IMAGE_DIR + ' - 需要多个类别的图片文件夹.')
        return -1

    # 图像几何变换
    (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(
        IMAGE_PROCESS_VALUE['flip_left_right'],
        IMAGE_PROCESS_VALUE['random_crop'],
        IMAGE_PROCESS_VALUE['random_scale'],
        IMAGE_PROCESS_VALUE['random_brightness'],
        MODEL_INFO['input_width'],
        MODEL_INFO['input_height'], MODEL_INFO['input_depth'],
        MODEL_INFO['input_mean'],  MODEL_INFO['input_std'])

    # download_and_extract(MODEL_INFO['data_url'])

    # 加载 inception_v3模型
    graph, bottleneck_tensor, jpeg_data_tensor = create_model_graph()

    with tf.Session(graph=graph) as sess:
        # 添加新的模型层
        # Add the new layer that we'll be training.
        (train_step, cross_entropy, bottleneck_input, ground_truth_input,
         final_tensor) = add_final_training_ops(
            len(image_lists.keys()),
            FINAL_TENSOR_NAME,
            bottleneck_tensor,
            MODEL_INFO['bottleneck_tensor_size'])

        init = tf.global_variables_initializer()
        sess.run(init)

        tf.keras.backend.set_session(sess)
        # save_graph_to_file(sess, graph, "./output/output_graph.pb")
        saver = tf.train.Saver()
        saver.save(tf.keras.backend.get_session(), "./output/saved_model")

        # new_model = tf.keras.experimental.load_from_saved_model("./output")
        tf.keras.experimental.export_saved_model(tf.keras.backend.get_session(), "./SavedModel")
