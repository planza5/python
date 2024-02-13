import numpy as np
import tensorflow as tf
from PIL import Image

# Función para cargar el modelo TFLite
def load_model(model_path):
    # Cargar el modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Función para procesar la imagen de entrada
def process_image(image_path, input_size):
    # Cargar imagen y redimensionar
    image = Image.open(image_path).convert('RGB')
    image = image.resize(input_size, Image.Resampling.LANCZOS)

    # Convertir a array y normalizar (para modelos cuantificados, escalar a 0-255 y convertir a uint8)
    image_array = np.array(image, dtype=np.uint8)
    return image_array



# Función para realizar la clasificación
def classify_image(interpreter, image):
    # Obtener información de entrada y salida del modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preparar la entrada del modelo
    interpreter.set_tensor(input_details[0]['index'], [image])

    # Realizar inferencia
    interpreter.invoke()

    # Obtener la salida del modelo
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Ruta del modelo y de la imagen
model_path = 'mobilenet_v1_1.0_224_quant.tflite'
image_path = 'gato.jpeg' # Cambia esto a la ruta de tu imagen

# Cargar el modelo
interpreter = load_model(model_path)

# Procesar la imagen
input_size = (224, 224) # Tamaño de entrada que espera el modelo
image = process_image(image_path, input_size)

# Clasificar la imagen
classification = classify_image(interpreter, image)

# Imprimir resultados de la clasificación
print("Resultados de la clasificación:", classification)
