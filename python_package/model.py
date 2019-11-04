# coding:utf-8

import tensorflow as tf
from core.model_process import create_model, add_new_last_layer, setup_to_transfer_learning
from core.image_process import create_image_lists
import os

from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api._v1.keras.applications.inception_v3 import preprocess_input


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ((x/255)-0.5)*2  归一化到±1之间
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)


def init():
    # 设置输出日志信息级别
    tf.logging.set_verbosity(tf.logging.INFO)

    train_dir = os.environ['IMAGE_TRAIN_INPUT_PATH']

    # 获取图片列表
    image_lists = create_image_lists(train_dir, 0, 0)

    # 检测文件夹是否符合训练的规则
    class_count = len(image_lists.keys())

    base_model, data_input = create_model()
    model = add_new_last_layer(base_model, class_count)

    setup_to_transfer_learning(model, base_model)

    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        target_size=(224, 224),  # Inception V3规定大小
                                                        batch_size=4)

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=800,  # 800
                        epochs=2,  # 2
                        validation_steps=12,  # 12
                        class_weight='auto'
                        )

    MODEL_PATH = os.environ['MODEL_INFERENCE_PATH']
    tf.keras.experimental.export_saved_model(model, MODEL_PATH + '/SavedModel')


init()
