# coding:utf-8
import tensorflow as tf
from constant import SUMMARIES_DIR, IMAGE_DIR, TESTING_PERCENTAGE, VALIDATION_PERCENTAGE
from core.model_process import create_model, add_new_last_layer, setup_to_transfer_learning, setup_to_fine_tune
from core.image_process import create_image_lists, add_input_distortions
from config import MODEL_INFO
import os

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.resnet50 import preprocess_input

import matplotlib.pyplot as plt


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


# 图片扭曲参数，即是几何变换
IMAGE_PROCESS_VALUE = {
    'flip_left_right': False,
    'random_crop': 0,
    'random_scale': 0,
    'random_brightness': 0
}

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ((x/255)-0.5)*2  归一化到±1之间
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(SUMMARIES_DIR)
    tf.gfile.MakeDirs(SUMMARIES_DIR)
    if(os.path.exists("SavedModel")):   # 判断文件夹是否存在
        os.removedirs("SavedModel")   # 删除文件夹
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

    base_model, data_input = create_model()

    train_generator = train_datagen.flow_from_directory(
        directory='./images',
        target_size=(224, 224),  # Inception V3规定大小
        batch_size=4
    )

    val_generator = val_datagen.flow_from_directory(
        directory='./validation-images',
        target_size=(224, 224),
        batch_size=4)

    model = add_new_last_layer(base_model,  class_count)

    setup_to_transfer_learning(model, base_model)

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=50,  # 800
                        epochs=2,  # 2
                        validation_data=val_generator,
                        validation_steps=12,  # 12
                        class_weight='auto'
                        )

    setup_to_fine_tune(model, base_model)

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=50,
                        epochs=2,
                        validation_data=val_generator,
                        validation_steps=1,
                        class_weight='auto'
                        )

    model.summary()

    tf.keras.experimental.export_saved_model(model, "./SavedModel")
