import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Carga el modelo MobileNetV2 pre-entrenado
modelo = MobileNetV2(weights='imagenet')
print(modelo.input_shape)
# Carga y prepara la imagen
# Reemplaza 'path_to_your_image.jpg' con la ruta a tu imagen
img_path = 'paisaje.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Realiza la predicción
predicciones = modelo.predict(img_array)

# Decodifica las predicciones
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predicciones, top=1)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
