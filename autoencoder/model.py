import tensorflow as tf
import numpy as np

# Build Model
class Model:
    '''
    model prototype
    '''
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
                    ground_truth = self.getNode(loss_config["ground_truth"])
                    prediction = self.getNode(loss_config["prediction"])
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

    def train(self):
        '''
        provide initialize operation ```init``` and training operation ```optimizer```
        '''
        # optimizer
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # init tf variables
        self.init = tf.global_variables_initializer()

    def getNode(self, path):
        # split path
        path = path.split('/')
        # find the node with the path
        node = self.nodes
        for config in path:
            node = node[config]
        return node

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
                    input_layer = self.getNode(layer_config["input"])
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

            # RNN
            elif layer_type == "RNN":
                # parameters
                output_size = layer_config["output_size"]
                # cell
                if "cell" not in layer_config or layer_config["cell"] is None:
                    # default RNN cell
                    cell = tf.contrib.rnn.BasicRNNCell(output_size, activation=layer_config["activation"])
                else:
                    # specified cell
                    cell = layer_config["cell"](output_size, activation=layer_config["activation"])

                # inputs & initial state
                if "input_mode" not in layer_config or layer_config["input_mode"] is None:
                    # TODO: zero input w/ shape [batch_size, time_step, data_size]
                    '''
                    # initial state
                    init_state = self.getNode(layer_config["init_state"])
                    # init state should be [batch, output_size]
                    batch_size = init_state.get_shape().as_list()[0]
                    # input size
                    if isinstance(layer_config["input"], int):
                        # specified fixed length
                        input_size = layer_config["input"]
                    elif isinstance(layer_config["input"], str):
                        # other layers parameter
                        input_size = self.getNode(layer_config["input"])
                    '''
                    pass
                    
                elif layer_config["input_mode"] == "INPUT_MODE":
                    # feed w/ input data
                    # input
                    input_layer = self.getNode(layer_config["input"])
                    # input data shape should be [batch_size, time_step, data_size]
                    batch_size = input_layer.get_shape().as_list()[0]
                    time_step = input_layer.get_shape().as_list()[1]
                    input_size = input_layer.get_shape().as_list()[2]
                    if "sequence_len" in layer_config and layer_config["sequence_len"] is not None:
                        if isinstance(layer_config["sequence_len"], int):
                            # specified fixed length
                            time_step = layer_config["sequence_len"]
                        elif isinstance(layer_config["sequence_len"], str):
                            # other layers parameter
                            time_step = self.getNode(layer_config["sequence_len"])
                    # initial state
                    if "init_state" in layer_config and layer_config["init_state"] is not None:
                        # init state should be [batch_size, output_size]
                        init_state = self.getNode(layer_config["init_state"])
                    else: 
                        # zero init state
                        #init_state = tf.zeros([batch_size, output_size], name="init_state")
                        init_state = cell.zero_state(batch_size, dtype=tf.float32)
                        print(init_state)

                    # build layer
                    with tf.variable_scope(layer_name):
                        _state = init_state
                        _outputs = []
                        # time major
                        input_layer = tf.transpose(input_layer, [1, 0, 2])
                        # recurrent
                        for step in range(time_step):
                            _output, _state = cell(input_layer[step], _state)
                            _outputs.append(_output)
                        # stack outputs [batch_size, time_step, data_size]
                        _outputs = tf.stack(_outputs, axis=1, name="outputs")
                
                elif layer_config["input_mode"] == "OUTPUT_MODE":
                    # feed w/ previous output
                    # initial state
                    init_state = self.getNode(layer_config["init_state"])
                    print(cell)
                    if isinstance(cell, tf.contrib.rnn.BasicLSTMCell):
                        print('wtf')
                        if not isinstance(init_state, tf.contrib.rnn.LSTMStateTuple):
                            print('wtff')
                            init_state = tf.contrib.rnn.LSTMStateTuple(init_state[1], init_state[0])
                    # init state should be [batch, output_size]
                    # TODO: dirty code
                    if hasattr(init_state, 'get_shape'):
                        # only one tensor
                        batch_size = init_state.get_shape().as_list()[0]
                    else:
                        # tuple of tensors
                        batch_size = init_state[0].get_shape().as_list()[0]
                    
                    # input size
                    if isinstance(layer_config["input"], int):
                        # specified fixed length
                        input_size = layer_config["input"]
                    elif isinstance(layer_config["input"], str):
                        # other layers parameter
                        input_size = self.getNode(layer_config["input"])
                    # time step
                    if isinstance(layer_config["sequence_len"], int):
                        # specified fixed length
                        time_step = layer_config["sequence_len"]
                    elif isinstance(layer_config["sequence_len"], str):
                        # other layers parameter
                        time_step = self.getNode(layer_config["sequence_len"])

                    # create FC for convert output from 
                    # [batch_size, output_size] to [batch_size, input_size]
                    if "fc_name" not in layer_config or layer_config["fc_name"] is None:
                        fc_name = "hidden2output"
                    else: 
                        fc_name = layer_config["fc_name"]
                    with tf.name_scope(fc_name):
                        # weight & bias
                        fc_weight = tf.Variable(
                            self.random_init([output_size, input_size]),
                            name="weight",
                        )
                        fc_bias = tf.Variable(
                            self.random_init([input_size]), name="bias"
                        )

                    # build layer
                    with tf.variable_scope(layer_name):
                        _state = init_state
                        _output = tf.zeros([batch_size, input_size])
                        _outputs = []
                        # fc activation
                        fc_activation = layer_config["fc_activation"]
                        if fc_activation is None:
                            fc_activation = tf.identity
                        # recurrent
                        for _ in range(time_step):
                            _output, _state = cell(_output, _state)
                            # FC
                            _output = fc_activation(
                                tf.matmul(_output, fc_weight) + fc_bias
                            )
                            _outputs.append(_output)
                        # stack outputs [batch_size, time_step, data_size]
                        _outputs = tf.stack(_outputs, axis=1, name="outputs")

                # register node
                self.nodes[layer_name] = {
                    "outputs": _outputs,
                    "state": _state,
                    "sequence_len": time_step,
                    "input_size": input_size
                }

            # Sampler for variational autoencoder
            elif layer_type == "sampler":
                with tf.name_scope(layer_name):
                    # TODO: only standard deviation for each dimension
                    #       might need to add convariance?
                    # TODO: activation function?

                    # input size
                    # input data shape should be [batch, data_size]
                    # TODO: reshape
                    input_layer = self.getNode(layer_config["input"])
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
                input_layer = self.getNode(layer_config["input"])
                # create identity
                self.nodes[layer_name] = tf.identity(input_layer, name=layer_name)

            # Block Input
            elif layer_type == "block_input":
                # input layer
                input_layer = self.getNode(layer_config["input"])
                # create placeholder with default input
                self.nodes[layer_name] = tf.placeholder_with_default(
                    input_layer, input_layer.get_shape(), name=layer_name
                )
            
            # Reshape TODO: as option for layers
            elif layer_type == "reshape":
                # input layer
                input_layer = self.getNode(layer_config["input"])
                # reshape
                self.nodes[layer_name] = tf.reshape(input_layer, layer_config["output_size"], name=layer_name)

            # Custom Function
            elif layer_type == "custom_function":
                with tf.name_scope(layer_name):
                    # input layer
                    inputs = []
                    if type(layer_config["input"]) == list:
                        for _input in layer_config:
                            inputs.append(self.getNode(_input))
                    else:
                        inputs.append(self.getNode(layer_config["input"]))
                    # custom function
                    self.nodes[layer_name] = layer_config["function"](inputs)
