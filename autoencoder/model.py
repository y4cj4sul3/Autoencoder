import tensorflow as tf
import numpy as np

# Build Model
class Model:
    def __init__(self, config):
        # Construct model
        # input
        self.input = tf.placeholder(tf.float32, config["input_shape"], name="input")

        # encoder
        with tf.name_scope("encoder"):
            self.encoder = self.createBlock(config["encoder"], self.input)

        # sampler
        if config["sampler"] == None:
            # fixed latent code
            self.decoder = self.encoder
        else:
            # TODO: variational AE
            # self.decoder =
            pass

        # decoder
        with tf.name_scope("decoder"):
            self.decoder = self.createBlock(config["decoder"], self.decoder)

        # output
        self.output = tf.identity(self.decoder, name="output")

        # loss
        # TODO: setting loss
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.pow(self.input - self.output, 2))

    def createBlock(self, block_config, input_layer):
        layers = input_layer
        for layer_config in block_config:
            if layer_config["type"] == "FC":
                # Fully Connected
                with tf.name_scope("FC"):
                    # weight & bias
                    weight = tf.Variable(
                        tf.random_normal(
                            [layer_config["input_size"], layer_config["output_size"]]
                        ),
                        name="weight",
                    )
                    bias = tf.Variable(
                        tf.random_normal([layer_config["output_size"]]), name="bias"
                    )
                    # build layer
                    layers = layer_config["activation"](
                        tf.matmul(layers, weight) + bias
                    )
        return layers

    def train(self, learning_rate):
        # optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        # init tf variables
        self.init = tf.global_variables_initializer()

