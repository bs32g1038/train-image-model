# coding:utf-8
import tensorflow as tf
from tensorflow.python.keras.api._v1.keras.preprocessing import image
import numpy as np
from tensorflow.python.keras.api._v1.keras.applications.inception_v3 import preprocess_input

img_path = "./none36.jpg"

img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(x)

model = tf.keras.experimental.load_from_saved_model('./SavedModel1')

preds = model.predict(x)

print(preds)
