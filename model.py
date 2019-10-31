# get the train images
train_dir = os.environ['IMAGE_TRAIN_INPUT_PATH']

#train the model
model = …
model.compile()
model.fit()
# or
model.fit_generator()

#save the model
tf.saved_model.simple_save(
    tf.keras.backend.get_session(), 
    MODEL_PATH + "/SavedModel/",
    inputs={"your_model_inputs": model.input},
    outputs={"your_model_outputs": model.output}
)