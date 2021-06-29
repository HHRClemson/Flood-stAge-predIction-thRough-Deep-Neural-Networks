import tensorflow as tf
from tensorflow.keras import layers, models, Model


class DCGAN(Model):

    def __init__(self, img_width=512, img_height=512, channels=1, **kwargs):
        super(DCGAN, self).__init__(name="DCGAN", **kwargs)
        # Train on 512x512 greyscale segmentation pictures (channel = 1)
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.generator: Model = self._create_generator()
        self.discriminator: Model = self._create_discriminator()

    @staticmethod
    def _create_generator() -> Model:
        generator = models.Sequential()
        generator.add(layers.Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))

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
        pass

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
    gan.generator.summary()