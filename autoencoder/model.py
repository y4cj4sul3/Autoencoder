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
        # nodes: every node or layer need register
        self.nodes = {}
        self.loss = 0

        # create blocks
        self.createBlocks(config["model"])

        # loss
        with tf.name_scope("loss"):
            for loss_config in config["loss"]:
                with tf.name_scope(loss_config["name"]):
                    # ground truth & prediction
                    ground_truth = self.nodes[loss_config["ground_truth"]]
                    prediction = self.nodes[loss_config["prediction"]]
                    # weight
                    loss_weight = loss_config["weight"]
                    if (
                        "loss_func" not in loss_config
                        or loss_config["loss_func"] is None
                    ):
                        # default loss: mean square error
                        loss = loss_weight * tf.reduce_mean(
                            tf.pow(ground_truth - prediction, 2)
                        )
                    else:
                        # custom loss
                        loss = loss_weight * loss_config["loss_func"](
                            ground_truth, prediction
                        )
                # add loss
                self.loss += loss
            # total loss (mean up batch dim)
            self.loss = tf.reduce_mean(self.loss)

    def train(self, learning_rate):
        # optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        # init tf variables
        self.init = tf.global_variables_initializer()

    def createBlocks(self, config):
        for block_config in config:
            # create block scope
            with tf.name_scope(block_config["name"]):
                # layers
                if "layers" in block_config and block_config["layers"] is not None:
                    # create layers
                    self.createLayers(block_config["layers"])
                # blocks
                if "blocks" in block_config and block_config["blocks"] is not None:
                    # create blocks
                    self.createBlocks(block_config["blocks"])

    def createLayers(self, config):
        for layer_config in config:
            # Config Parameters
            layer_type = layer_config["type"]
            layer_name = layer_config["name"]
            # Create Layer
            # Fully Connected
            if layer_type == "FC":
                with tf.name_scope(layer_name):
                    # input size
                    # input data shape should be [batch, data_size]
                    # TODO: reshape
                    input_layer = self.nodes[layer_config["input"]]
                    input_size = input_layer.get_shape().as_list()[1]
                    # weight & bias
                    weight = tf.Variable(
                        self.random_init([input_size, layer_config["output_size"]]),
                        name="weight",
                    )
                    bias = tf.Variable(
                        self.random_init([layer_config["output_size"]]), name="bias"
                    )
                    # build layer
                    activation = layer_config["activation"]
                    if activation is None:
                        # linear
                        self.nodes[layer_name] = tf.matmul(input_layer, weight) + bias
                    else:
                        self.nodes[layer_name] = activation(
                            tf.matmul(input_layer, weight) + bias
                        )

            # Vanilla RNN
            elif layer_type == "RNN":
                """
                # weight & bias
                weight = tf.Variable(
                    self.random_init(
                        [layer_config["input_size"] + layer_config["output_size"], layer_config["output_size"]]
                    ),
                    name="weight",
                )
                bias = tf.Variable(
                    self.random_init([layer_config["output_size"]]), name="bias"
                )
                # build layer
                for _ in range(layer_config["sequence_len"]):
                    # input
                    if layer_config["input"] == "new input":
                        # create new placehold
                        pass
                    else:
                        # get previous layer
                """
                pass

            # LSTM
            elif layer_type == "LSTM":
                """
                # parameters
                hidden_size = layer_config["output_size"]
                forget_bias = layer_config["forget_bias"]
                batch_size
                # create BasicLSTMCell
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=forget_bias, state_is_tuple=True)
                # defining initial state
                init_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
                outputs, states = rnn.dynamic_rnn(lstm_cell, layers)
                """
                pass

            # Sampler for variational autoencoder
            elif layer_type == "sampler":
                with tf.name_scope(layer_name):
                    # TODO: only standard deviation for each dimension
                    #       might need to add convariance?
                    # TODO: activation function?

                    # input size
                    # input data shape should be [batch, data_size]
                    # TODO: reshape
                    input_layer = self.nodes[layer_config["input"]]
                    input_size = input_layer.get_shape().as_list()[1]
                    # mean
                    with tf.name_scope("mean"):
                        # weight & bias
                        z_mean_w = tf.Variable(
                            self.random_init([input_size, layer_config["output_size"]]),
                            name="weight",
                        )
                        z_mean_b = tf.Variable(
                            self.random_init([layer_config["output_size"]]), name="bias"
                        )
                        # build vector
                        z_mean = tf.matmul(input_layer, z_mean_w) + z_mean_b
                    # standard deviation (actually is 4*log(std)?)
                    with tf.name_scope("standard_deviation"):
                        # weight & bias
                        z_std_w = tf.Variable(
                            self.random_init([input_size, layer_config["output_size"]]),
                            name="weight",
                        )
                        z_std_b = tf.Variable(
                            self.random_init([layer_config["output_size"]]), name="bias"
                        )
                        # build vector
                        z_std = tf.matmul(input_layer, z_std_w) + z_std_b
                    # epsilon
                    epsilon = tf.random_normal(
                        [layer_config["output_size"]],
                        dtype=tf.float32,
                        mean=0.,
                        stddev=1.0,
                        name="epsilon",
                    )
                    # reparameterize trick
                    # z = mean + var*eps
                    z = z_mean + tf.exp(z_std / 2) * epsilon
                    self.nodes[layer_name] = z

                    # loss
                    with tf.name_scope("KL_divergence_loss"):
                        loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
                        # TODO: setting beta (currentlt 0.5)
                        self.loss += -0.5 * tf.reduce_sum(loss, 1)

            # Input
            elif layer_type == "input":
                # create placeholder
                self.nodes[layer_name] = tf.placeholder(
                    tf.float32, layer_config["shape"], name=layer_name
                )

            # Output
            elif layer_type == "output":
                # input layer
                input_layer = self.nodes[layer_config["input"]]
                # create identity
                self.nodes[layer_name] = tf.identity(input_layer, name=layer_name)

            # Block Input
            elif layer_type == "block_input":
                # input layer
                input_layer = self.nodes[layer_config["input"]]
                # create placeholder with default input
                self.nodes[layer_name] = tf.placeholder_with_default(
                    input_layer, input_layer.get_shape(), name=layer_name
                )
