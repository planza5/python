import tensorflow as tf
from tensorflow.keras import layers, models

latent_dim = 100
epochs = 500
batch = 32

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
    gan_input = tf.keras.Input(shape=(latent_dim,))  # Define la dimensión del espacio latente
    generated_image = generator(gan_input)  # Genera la imagen a partir del ruido
    gan_output = discriminator(generated_image)  # Discrimina la imagen generada

    # Crear el modelo GAN con el input del generador y el output del discriminador
    gan = models.Model(gan_input, gan_output)

    # Compilar el modelo GAN
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    return gan

import numpy as np

def get_real_images(X_train, num_samples=16):
    # Seleccionar índices de manera aleatoria
    indices = np.random.randint(0, X_train.shape[0], num_samples)
    # Seleccionar las imágenes
    images = X_train[indices]
    # Normalizar las imágenes al rango [-1, 1]
    images_normalized = (images.astype('float32') / 127.5) - 1
    return images_normalized


def get_fake_images(generator, latent_dim, num_images=16):
    # Generar vectores de ruido aleatorio
    noise = np.random.normal(0, 1, (num_images, latent_dim))

    # Generar imágenes a partir del ruido
    generated_images = generator.predict(noise)

    return generated_images


# carga datos
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Crea el generador
generator = build_generator(latent_dim)

# Crea el disrciminador y compilamos
discriminator = build_discriminator(image_shape=(28, 28, 1))
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gan = build_gan(generator,discriminator)

for epoch in range(epochs):
    real_images = get_real_images(x_train,16)
    real_labels = np.ones((16, 1))

    fake_images = get_fake_images(generator,latent_dim)
    fake_labels = np.zeros((16, 1))

    # Entrenar el discriminador
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch, latent_dim))
    misleading_labels = np.ones((batch, 1))  # Etiquetas engañosas para el generador

    g_loss = gan.train_on_batch(noise, misleading_labels)

    print(f"Epoch: {epoch + 1}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")