# https://www.tensorflow.org/lite/convert/#convert_a_savedmodel_recommended_

import tensorflow as tf

# Convert the model
saved_model_dir = "./savedModels/sign_asl_personal01"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)