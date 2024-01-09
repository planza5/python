import tensorflow as tf

# Carga el modelo preentrenado de ResNet50
model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))

# Convertir el modelo a formato TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo convertido
with open('resnet50.tflite', 'wb') as f:
    f.write(tflite_model)
