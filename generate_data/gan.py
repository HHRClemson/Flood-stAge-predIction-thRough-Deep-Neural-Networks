import numpy as np
import time

import tensorflow as tf
from tensorflow.keras import layers, Model


class ReGAN(Model):

    def __init__(self,
                 image_size=256,
                 channels=3,
                 batch_size=32,
                 noise_dim=2048,
                 kernel_size=3,
                 stride_size=2,
                 **kwargs):
        super(ReGAN, self).__init__(name="ReGAN", **kwargs)

        self.image_size = image_size
        self.channels = channels
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size

        self.generator: Model = self._create_generator()
        self.generator.summary()

        self.discriminator = self._create_discriminator()
        self.discriminator.summary()

    def _create_generator(self) -> Model:
        """
        Create the generator which generates a new image based on the given gauge height
        """
        latent_space_input = layers.Input(shape=(self.noise_dim, ))
        gauge_height_input = layers.Input(shape=(1,))

        """
        We use the naive label input mechanism (NLI) by Ding et al. 
        for a continuous conditional GAN (CcGAN).
        See here: https://arxiv.org/pdf/2011.07466.pdf
        """
        x = tf.add(latent_space_input, gauge_height_input)

        # Reshape to be valid input for the Conv2DTranspose layer
        x = layers.Reshape(target_shape=[1, 1, self.noise_dim], input_shape=[self.noise_dim])(x)
        x = layers.Conv2DTranspose(filters=512, kernel_size=2)(x)
        x = layers.Activation("relu")(x)

        filters = [4, 8, 16, 32, 64, 128, 256]
        for f in reversed(filters):
            x = layers.Conv2D(filters=f, kernel_size=self.kernel_size, padding="same")(x)
            x = layers.BatchNormalization(momentum=0.7)(x)
            x = layers.Activation("relu")(x)
            x = layers.UpSampling2D()(x)

        x = layers.Conv2D(filters=self.channels, kernel_size=self.kernel_size, padding="same")(x)
        x = layers.Activation("tanh")(x)

        return Model(inputs=[latent_space_input, gauge_height_input],
                     outputs=[x],
                     name="Generator")

    def _create_discriminator(self) -> Model:
        """
        Create a discriminator to judge the created image with it corresponding gauge height.
        """
        image_input = layers.Input(shape=(self.image_size, self.image_size, self.channels))
        gauge_height_input = layers.Input(shape=(1,))

        x = image_input
        filters = [4, 8, 16, 32, 64, 128, 256]
        for f in filters:
            x = layers.Conv2D(filters=f, kernel_size=3, padding="same")(x)
            x = layers.BatchNormalization(momentum=0.7)(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Dropout(0.25)(x)
            x = layers.AveragePooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dense(1)(x)

        """
        We use the naive label input mechanism (NLI) proposed by Ding et al. in 2021
        for a continuous conditional GAN (CcGAN).
        See here: https://arxiv.org/pdf/2011.07466.pdf
        """
        inner_product = tf.tensordot(x, gauge_height_input, axes=1)
        x = layers.Dense(1)(x)
        out = tf.add(x, inner_product)

        return Model(inputs=[image_input, gauge_height_input],
                     outputs=[out],
                     name="Discriminator")

    def train(self, dataset, epochs, batch_size):
        generated_images = []

        for epoch in range(epochs):
            start = time.time()

            next_batch = []
            min_gen_loss = float("inf")
            min_disc_loss = float("inf")

            for i in range(len(dataset)):
                next_batch.append(dataset[i])
                if i % batch_size == 0:
                    gen_loss, disc_loss = self.train_step(np.array(next_batch))
                    min_gen_loss = min(min_gen_loss, gen_loss)
                    min_disc_loss = min(min_disc_loss, disc_loss)
                    next_batch = []

            # generate 10 times pictures based on the generator trained to this epoch
            # to visualize the improvements and find the sweet epoch point
            if epoch % (epochs / 10) == 0:
                generated_images.append(self.call(tf.random.normal([16, self.noise_dim])))

            print('Time for epoch {e}/{ne} is {t} sec. Generator loss: {gl}, Discriminiator loss: {dl}'
                  .format(e=epoch + 1,
                          ne=epochs,
                          t=time.time() - start,
                          gl=min_gen_loss,
                          dl=min_disc_loss))

        return generated_images

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            generator_loss = self.gen_loss(fake_output)
            discriminator_loss = self.disc_loss(real_output, fake_output)

        gen_grads = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        disc_grads = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return generator_loss, discriminator_loss

    def call(self, inputs, training=None, mask=None):
        for img, gauge_height in inputs:
            encoded = self.encoder(img)
            yield self.generator(encoded, gauge_height)

    def get_config(self):
        config = super(ReGAN, self).get_config()
        config.update({
            "image_size": self.image_size,
            "channels": self.channels,
            "latent_dim": self.latent_dim,
            "batch_size": self.batch_size,
            "kernel_size": self.kernel_size
        })
        return config

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.generator.summary(line_length, positions, print_fn)
        self.discriminator.summary(line_length, positions, print_fn)


if __name__ == "__main__":
    gan = ReGAN()
    #gan.summary()
