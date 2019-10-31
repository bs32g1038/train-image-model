from zoo.common.nncontext import *
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *
from zoo.pipeline.api.keras.layers import Dense, Input, Flatten
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.net import *
from bigdl.optim.optimizer import *

sc = init_nncontext("train keras")
img_path = "./image"
image_set = ImageSet.read(img_path, sc, min_partitions=1)
transformer = ChainedPreprocessing([
    ImageResize(256, 256),
    ImageCenterCrop(224, 224),
    ImageChannelNormalize(123.0, 117.0, 104.0),
    ImageMatToTensor(),
    ImageSetToSample()
])
image_data = transformer(image_set)
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
label_rdd = sc.parallelize(labels, 1)
samples = image_data.get_image().zip(label_rdd).map(
    lambda tuple: Sample.from_ndarray(tuple[0], tuple[1]))
# create model
model_path = "./zoo_mobilenet.model"
full_model = Net.load_bigdl(model_path)
# create a new model by remove layers after pool5/drop_7x7_s1
model = full_model.new_graph(["pool5/drop_7x7_s1"])
# freeze layers from input to pool4/3x3_s2 inclusive
model.freeze_up_to(["pool4/3x3_s2"])

inputNode = Input(name="input", shape=(3, 224, 224))
inception = model.to_keras()(inputNode)
flatten = Flatten()(inception)
logits = Dense(2)(flatten)
lrModel = Model(inputNode, logits)

batchsize = 4
nEpochs = 10
lrModel.compile(optimizer=Adam(learningrate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
lrModel.fit(x=samples, batch_size=batchsize, nb_epoch=nEpochs)
