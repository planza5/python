import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

print("Dispositivos físicos disponibles:", tf.config.list_physical_devices())

#Intenta realizar una operación simple usando TensorFlow
tf.debugging.set_log_device_placement(True)

# Crea algunos tensores
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
model = ResNet50(weights='imagenet')
img_path = 'loro.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

print(np.array(img_array).size)


predictions = model.predict(img_array)
print('Predicciones:', decode_predictions(predictions, top=3)[0])

