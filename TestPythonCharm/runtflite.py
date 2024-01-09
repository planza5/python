from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def obtener_clase(indice, archivo='imagenet_labels.txt'):
    try:
        with open(archivo, 'r') as file:
            clases = file.readlines()
        # Los índices en Python comienzan en 0, así que no es necesario ajustar el índice
        return clases[indice].strip()  # .strip() elimina los espacios en blanco y saltos de línea al principio y final de la línea
    except IndexError:
        return "Índice fuera de rango."
    except FileNotFoundError:
        return "Archivo no encontrado."

def load_and_prepare_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    new_img_array = preprocess_input(img_array)

    return new_img_array




# Cargar el modelo
interpreter = tf.lite.Interpreter(model_path="mobilenetv2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Realizar la inferencia
interpreter.set_tensor(input_details[0]['index'], load_and_prepare_image('paisaje.jpg'))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

predicted_class = np.argmax(output_data[0])

# Imprimir el índice de la clase predicha
print("Clase predicha:", obtener_clase(predicted_class))