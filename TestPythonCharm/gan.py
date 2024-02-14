import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt


def build_generator(latent_dim):
    model = tf.keras.Sequential()

    # Comienza con un densamente conectado y cambia a las dimensiones que deseas para la imagen, p.ej., 7x7x128
    model.add(layers.Dense(7 * 7 * 128, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Cambia a la forma de imagen que es 7x7x128
    model.add(layers.Reshape((7, 7, 128)))

    # Convolución transpuesta para 14 x 14
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Otra convolución transpuesta para aumentar a 28x28, el tamaño de las imágenes MNIST
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Capa de salida para generar la imagen final de 28x28 con 1 canal (escala de grises)
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model


def build_discriminator(image_shape):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def build_gan(generator, discriminator):
    # Asegurar que el discriminador no se actualice durante el entrenamiento del generador dentro de la GAN
    discriminator.trainable = False

    # Modelo compuesto
    gan_input = tf.keras.Input(shape=(config.latent_dim,))  # Define la dimensión del espacio latente
    generated_image = generator(gan_input)  # Genera la imagen a partir del ruido
    gan_output = discriminator(generated_image)  # Discrimina la imagen generada

    # Crear el modelo GAN con el input del generador y el output del discriminador
    gan = models.Model(gan_input, gan_output)

    return gan


def get_fake_images(generator, latent_dim, num_images=16):
    # Generar vectores de ruido aleatorio
    noise = np.random.normal(0, 1, (num_images, latent_dim))

    # Generar imágenes a partir del ruido
    generated_images = generator.predict(noise)

    return generated_images


wandb.init(project='MNIST Santander', entity='pplanza2')

config = wandb.config
config.learning_rate_dis = 0.0001
config.learning_rate_gen = 0.001
config.epochs=1000
config.latent_dim = 100
config.batch_size = 32


# carga datos
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Configuración del optimizador con la tasa de aprendizaje especificada
adam_optimizer_dis = tf.keras.optimizers.Adam(learning_rate=config.learning_rate_dis)
adam_optimizer_gen = tf.keras.optimizers.Adam(learning_rate=config.learning_rate_gen)

# Crea el generador
generator = build_generator(config.latent_dim)

# Crea el disrciminador y compilamos
discriminator = build_discriminator(image_shape=(28, 28, 1))
discriminator.compile(optimizer=adam_optimizer_dis, loss='binary_crossentropy', metrics=['accuracy'])

gan = build_gan(generator,discriminator)
gan.compile(optimizer=adam_optimizer_gen, loss='binary_crossentropy')

# Suponiendo que `x_train` es tu conjunto de datos de imágenes.
num_batches = int(np.ceil(x_train.shape[0] / float(config.batch_size)))

for epoch in range(config.epochs):
    for batch_num in range(num_batches):
        #indices inicio y fin segun batch
        start_idx = batch_num * config.batch_size
        end_idx = min((batch_num + 1) * config.batch_size, x_train.shape[0])

        # Traemos lote de imagenes reales y etiquetas a 1
        real_images = x_train[start_idx:end_idx]
        real_images = (real_images.astype('float32') / 127.5) - 1
        real_images = np.expand_dims(real_images, axis=-1)
        real_labels = np.ones((config.batch_size, 1))

        # Traemos lote de imagenes falsas y etiquetas a 0
        fake_images = get_fake_images(generator,config.latent_dim, config.batch_size)
        fake_labels = np.zeros((config.batch_size, 1))

        # Entrenar el discriminador
        if batch_num == 0 or batch_num % 3 == 0:
            print("Entrenando discriminador")
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        # Promedio de las pérdidas
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Ruido y etiquetas falsas para engañar al discriminador
        noise = np.random.normal(0, 1, (32, config.latent_dim))
        misleading_labels = np.ones((32, 1))  # Etiquetas engañosas para el generador

        # Entrenando generador
        print("Entrenando generador")
        g_loss = gan.train_on_batch(noise, misleading_labels)

        wandb.log({'epoch': epoch, 'd_loss': d_loss[0], 'g_loss': g_loss, 'd_acc': d_loss[1]})
        print(f"Epoch: {epoch + 1}/{config.epochs}, batch nº: {batch_num}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

        if epoch % 100 == 0:  # Ejemplo: Cada 10 épocas
            # Generar imágenes de muestra
            sample_images = get_fake_images(generator, config.latent_dim, num_images=config.batch_size)
            # Escalar las imágenes del rango [-1, 1] al rango [0, 255]
            sample_images = ((sample_images * 0.5) + 0.5) * 255.0
            # Asegúrate de que las imágenes sean enteros, ya que los píxeles deben ser valores enteros
            sample_images = np.clip(sample_images, 0, 255).astype(np.uint8)
            wandb.log({"examples": [wandb.Image(img, caption=f"Epoch {epoch}") for img in sample_images]})
            wandb.log({"reals": [wandb.Image(img, caption=f"Epoch {epoch}") for img in real_images]})