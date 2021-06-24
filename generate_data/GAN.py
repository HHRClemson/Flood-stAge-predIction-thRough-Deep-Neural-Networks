
import tensorflow as tf
from tensorflow.keras import models, layers, Model


class GAN(Model):

    def __init__(self, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.generator = self._create_generator()
        self.discriminator = self._create_discriminator()

    @staticmethod
    def _create_generator() -> Model:
        generator = models.Sequential()
        return generator

    @staticmethod
    def _create_discriminator() -> Model:
        discriminator = models.Sequential()
        return discriminator

    @tf.function
    def train_step(self, images):
        pass
