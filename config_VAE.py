import tensorflow as tf

# Network Parameters
image_dim = 784  # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

# custom loss function (reconstrution loss only)
def custom_loss(ground_true, prediction):
    loss = ground_true * tf.log(1e-10 + prediction) + (1 - ground_true) * tf.log(
        1e-10 + 1 - prediction
    )
    loss = -tf.reduce_sum(loss, 1)
    return loss


# custom random initializer
def custom_random_init(shape, name=None):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.), name=name)


# Model Architecture Configure
config = {
    "input_shape": [None, image_dim],
    "encoder": [
        {
            "type": "FC",
            "input_size": image_dim,
            "output_size": hidden_dim,
            "activation": tf.nn.tanh,
        }
    ],
    "sampler": {"input_size": hidden_dim, "output_size": latent_dim},
    "decoder": [
        {
            "type": "FC",
            "input_size": latent_dim,
            "output_size": hidden_dim,
            "activation": tf.nn.tanh,
        },
        {
            "type": "FC",
            "input_size": hidden_dim,
            "output_size": image_dim,
            "activation": tf.nn.sigmoid,
        },
    ],
    "loss": custom_loss,
    "random_init": custom_random_init,
}
