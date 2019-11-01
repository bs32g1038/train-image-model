# coding:utf-8
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

InceptionV3 = keras.applications.inception_v3.InceptionV3
preprocess_input = keras.applications.inception_v3.preprocess_input

# tf.keras.backend.set_session(tf.Session())

print(tf.__version__)

labels_nums = 5  # 类别个数
batch_size = 4  #
resize_height = 224  # 指定存储图片高度
resize_width = 224  # 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

# 定义input_images为图片数据
input_images = tf.placeholder(
    dtype=tf.float32,
    shape=[None, resize_height, resize_width, depths],
    name='input')

# def train(args):
#     """Use transfer learning and fine-tuning to train a network on a new dataset"""
#     train_img = 'training_set/'
#     validation_img = 'test_set/'
#     nb_epoch = int(args.nb_epoch)
#     nb_train_samples = get_nb_files(train_img)
#     nb_classes = len(glob.glob(train_img + "/*"))
#     # data prep
#     train_datagen = ImageDataGenerator(rotation_range=40,
#                                        width_shift_range=0.2,
#                                        height_shift_range=0.2,
#                                        rescale=1. / 255,
#                                        shear_range=0.2,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True,
#                                        fill_mode='nearest')

#     validation_datagen = ImageDataGenerator(rotation_range=40,
#                                             width_shift_range=0.2,
#                                             height_shift_range=0.2,
#                                             rescale=1. / 255,
#                                             shear_range=0.2,
#                                             zoom_range=0.2,
#                                             horizontal_flip=True,
#                                             fill_mode='nearest')

#     train_generator = train_datagen.flow_from_directory(
#         train_img,
#         target_size=(299, 299),
#         batch_size=32,
#         class_mode='categorical')
#     validation_generator = validation_datagen.flow_from_directory(
#         validation_img,
#         target_size=(299, 299),
#         batch_size=32,
#         class_mode='categorical')
#     if (K.image_dim_ordering() == 'th'):
#         input_tensor = Input(shape=(3, 299, 299))
#     else:
#         input_tensor = Input(shape=(299, 299, 3))

#     # setup model
#     base_model = InceptionV3(
#         input_tensor=input_tensor, weights='imagenet',
#         include_top=False)  #include_top=False excludes final FC layer
#     model = add_new_last_layer(base_model, nb_classes)

#     # transfer learning
#     setup_to_transfer_learn(model, base_model)

#     history_tl = model.fit_generator(train_generator,
#                                      samples_per_epoch=320,
#                                      nb_epoch=nb_epoch,
#                                      validation_data=validation_generator,
#                                      nb_val_samples=64)
#     model.save(args.output_model_file)
#     if args.plot:
#         plot_training(history_tl)

base_model = InceptionV3(input_tensor=input_images,
                         weights='imagenet',
                         include_top=False)

base_model.compile(optimizer=tf.train.AdamOptimizer(.00002),
                   loss=tf.keras.losses.MeanSquaredError(),
                   metrics=['accuracy'])

tf.keras.experimental.export_saved_model(base_model, "./SavedModel")