import tensorflow as tf
import numpy as np

# Build Model
class Model:
    def __init__(self, config):
        # Settings
        # random initializer
        if "random_init" in config and config["random_init"] is not None:
            # custom random initializer
            self.random_init = config["random_init"]
        else:
            # default: normal distribution
            self.random_init = tf.random_normal

        # Construct model
        # input
        self.input = tf.placeholder(tf.float32, config["input_shape"], name="input")

        # encoder
        with tf.name_scope("encoder"):
            # encoder
            self.encoder = self.createBlock(config["encoder"], self.input)

            # sampler
            if "sampler" in config and config["sampler"] is not None:
                # variational autoencoder
                with tf.name_scope("sampler"):
                    self.encoder = self.createSampler(config["sampler"], self.encoder)

        # decoder
        with tf.name_scope("decoder"):
            # decoder input
            self.decoder = tf.placeholder_with_default(
                self.encoder, self.encoder.get_shape(), name="decoder_input"
            )
            # decoder
            self.decoder = self.createBlock(config["decoder"], self.decoder)

        # output
        self.output = tf.identity(self.decoder, name="output")

        # loss
        with tf.name_scope("loss"):
            # reconstruction loss
            with tf.name_scope("reconstruction_loss"):
                if "loss" not in config or config["loss"] is None:
                    # default loss: mean square error
                    self.encode_decode_loss = tf.reduce_mean(
                        tf.pow(self.input - self.output, 2)
                    )
                else:
                    # custom loss
                    self.encode_decode_loss = config["loss"](self.input, self.output)

            # variational loss
            if "sampler" not in config or config["sampler"] is None:
                self.loss = self.encode_decode_loss
            else:
                # KL divergence
                with tf.name_scope("KL_divergence_loss"):
                    self.kl_div_loss = (
                        1 + self.z_std - tf.square(self.z_mean) - tf.exp(self.z_std)
                    )
                    self.kl_div_loss = -0.5 * tf.reduce_sum(self.kl_div_loss, 1)
                self.loss = tf.reduce_mean(self.encode_decode_loss + self.kl_div_loss)

    def train(self, learning_rate):
        # optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        # init tf variables
        self.init = tf.global_variables_initializer()

    def createBlock(self, block_config, input_layer):
        layers = input_layer
        for layer_config in block_config:
            if layer_config["type"] == "FC":
                # Fully Connected
                with tf.name_scope("FC"):
                    # weight & bias
                    weight = tf.Variable(
                        self.random_init(
                            [layer_config["input_size"], layer_config["output_size"]]
                        ),
                        name="weight",
                    )
                    bias = tf.Variable(
                        self.random_init([layer_config["output_size"]]), name="bias"
                    )
                    # build layer
                    layers = layer_config["activation"](
                        tf.matmul(layers, weight) + bias
                    )
        return layers

    def createSampler(self, block_config, input_layer):
        # mean
        with tf.name_scope("mean"):
            # weight & bias
            z_mean_w = tf.Variable(
                self.random_init(
                    [block_config["input_size"], block_config["output_size"]]
                ),
                name="weight",
            )
            z_mean_b = tf.Variable(self.random_init([block_config["output_size"]]))
            # build vector
            self.z_mean = tf.matmul(self.encoder, z_mean_w) + z_mean_b
        # standard deviation
        with tf.name_scope("standard_deviation"):
            # weight & bias
            z_std_w = tf.Variable(
                self.random_init(
                    [block_config["input_size"], block_config["output_size"]]
                )
            )
            z_std_b = tf.Variable(self.random_init([block_config["output_size"]]))
            # build vector
            self.z_std = tf.matmul(self.encoder, z_std_w) + z_std_b
        # epsilon
        epsilon = tf.random_normal(
            [block_config["output_size"]],
            dtype=tf.float32,
            mean=0.,
            stddev=1.0,
            name="epsilon",
        )
        # z = mean + std*eps
        z = self.z_mean + tf.exp(self.z_std / 2) * epsilon
        return z
