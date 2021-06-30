import tensorflow as tf
from tensorflow.keras import layers, models, Model


class DCGAN(Model):

    def __init__(self, img_width=512, img_height=512, channels=1, noise_dim=4096, batch_size=128, **kwargs):
        super(DCGAN, self).__init__(name="DCGAN", **kwargs)
        # Train on 512x512 greyscale segmentation pictures (channel = 1)
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.noise_dim = noise_dim
        self.batch_size = batch_size

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator: Model = self._create_generator()
        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        # Compare the decision of the discriminator on the generated image to an array of 1s
        self.gen_loss = lambda fake_out: cross_entropy(tf.ones_like(fake_out), fake_out)

        self.discriminator: Model = self._create_discriminator()
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)
        # Compare the discriminator decision of the real images to an array of 1s and the
        # discriminator predictions on the fake images to an array of 0s
        self.disc_loss = lambda real_out, fake_out: sum([
            cross_entropy(tf.ones_like(real_out), real_out),
            cross_entropy(tf.zeros_like(fake_out), fake_out)
        ])

    def _create_generator(self) -> Model:
        generator = models.Sequential()
        generator.add(layers.Reshape(
            target_shape=[1, 1, self.noise_dim], input_shape=[self.noise_dim]))

        generator.add(layers.Conv2DTranspose(filters=512, kernel_size=4))
        generator.add(layers.Activation("relu"))

        filters = [4, 8, 16, 32, 64, 128, 256]
        for f in reversed(filters):
            print(f)
            generator.add(layers.Conv2D(filters=f, kernel_size=3, padding="same"))
            generator.add(layers.BatchNormalization(momentum=0.7))
            generator.add(layers.Activation("relu"))
            generator.add(layers.UpSampling2D())

        generator.add(layers.Conv2D(filters=1, kernel_size=3, padding="same"))
        generator.add(layers.Activation("relu"))

        return generator

    def _create_discriminator(self) -> Model:
        discriminator = models.Sequential()
        discriminator.add(layers.Input(shape=(self.img_width, self.img_height, self.channels)))

        filters = [4, 8, 16, 32, 64, 128, 256]
        for f in filters:
            discriminator.add(layers.Conv2D(filters=f, kernel_size=3, padding="same"))
            discriminator.add(layers.BatchNormalization(momentum=0.7))
            discriminator.add(layers.LeakyReLU(0.2))
            discriminator.add(layers.Dropout(0.25))
            discriminator.add(layers.AveragePooling2D())

        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(128))
        discriminator.add(layers.LeakyReLU(0.2))
        discriminator.add(layers.Dense(1))

        return discriminator

    def _judge_gauge_height_prediction(self):
        pass

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

    def call(self, inputs, training=None, mask=None):
        output = []
        for image in inputs:
            output.append(self.generator(image))
        return output

    def get_config(self):
        config = super(DCGAN, self).get_config()
        config.update({"img_width": self.img_width, "img_height": self.img_height})
        return config

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.generator.summary(line_length, positions, print_fn)
        self.discriminator.summary(line_length, positions, print_fn)


if __name__ == "__main__":
    gan = DCGAN()
    gan.summary()
