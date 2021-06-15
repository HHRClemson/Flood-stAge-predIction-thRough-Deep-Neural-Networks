import tensorflow as tf
from tensorflow.keras import models, Model, layers
import tensorflow_probability as tfp


# Create a bayesian CNN to predict a range and a mean on a standard distribution
# for a regression task
def create_bcnn_model(train_size, img_shapes) -> Model:
    bcnn = models.Sequential()
    bcnn.add(layers.Conv2D(32, 3, 3, activation="relu", input_shape=img_shapes))
    bcnn.add(layers.MaxPool2D((2, 2)))
    bcnn.add(layers.Conv2D(64, 3, 3, activation="relu"))
    bcnn.add(layers.MaxPooling2D(2, 2))
    bcnn.add(layers.Conv2D(64, 3, 3, activation="relu"))

    bcnn.add(layers.GlobalMaxPooling2D())

    hidden_units = [128, 64]
    for units in hidden_units:
        bcnn.add(tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="relu",
        ))

    bcnn.add(layers.Dense(1, activation="linear"))

    bcnn.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    return bcnn


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model
