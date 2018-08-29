import tensorflow as tf
import copy

# Network Parameters
batch_size = 128
data_size = 1
max_time_step = 10
num_hidden = 12

# custom function
def teacher_forcing_input(inputs):
    # input shape should be [batch_size, time_step, data_size]
    _input = inputs[0]
    # get zero padding shape
    pad_shape = _input.get_shape().as_list()
    pad_shape[1] = 1
    # pad at t=0 and remove t=T
    return tf.concat([tf.zeros(pad_shape), _input[:, :-1]], axis=1)

# training
config_train = {
    "model": [
        {
            "name": "inputs",
            "layers": [
                {
                    "type": "input",
                    "name": "input",  # input name
                    "shape": [batch_size, max_time_step, data_size],  # input shape
                }
            ],
        },
        {
            "name": "encoder",
            "layers": [
                {
                    "type": "RNN",
                    "name": "encoder",
                    "input": "input",
                    "input_mode": "INPUT_MODE",  # input, zeros, output
                    "init_state": None,  # init state
                    "cell": tf.contrib.rnn.BasicLSTMCell,
                    "output_size": num_hidden,  # i.e. hidden state size
                    "activation": None,
                    "sequence_len": max_time_step,  # recurrent len
                }
            ],
        },
        {
            "name": "decoder",
            "layers": [
                {
                    "type": "custom_function",
                    "name": "teacher_forcing_input",
                    "input": "input",
                    "function": teacher_forcing_input
                },
                {
                    "type": "RNN",
                    "name": "RNN_section",
                    "input": "teacher_forcing_input",
                    "input_mode": "INPUT_MODE",
                    "init_state": "encoder/state",
                    "cell": tf.contrib.rnn.BasicLSTMCell,
                    "output_size": num_hidden,
                    "activation": None,
                    "sequence_len": max_time_step,  # as encoder
                    "fc_activation": None,
                },
                {
                    "type": "reshape",
                    "name": "flatten",
                    "input": "RNN_section/outputs",
                    "output_size": [-1, num_hidden],
                },
                {
                    "type": "FC",
                    "name": "hidden2output",
                    "input": "flatten",
                    "output_size": data_size,
                    "activation": None
                },
                {
                    "type": "reshape",
                    "name": "output",
                    "input": "hidden2output",
                    "output_size": [batch_size, max_time_step, data_size]
                }
            ],
        },
    ],
    "loss": [
        {
            "name": "enc_dec_loss",
            "weight": 1,
            "ground_truth": "input",
            "prediction": "output",
        }
    ],
}

# evaluation
config_eval = copy.deepcopy(config_train)
# decoder
config_eval["model"][2]["layers"] = [
    {
        "type": "RNN",
        "name": "RNN_section",
        "input": "encoder/input_size",  # specify data size
        "input_mode": "OUTPUT_MODE",
        "init_state": "encoder/state",
        "cell": tf.contrib.rnn.GRUCell,
        "output_size": num_hidden,
        "activation": None,
        "sequence_len": "encoder/sequence_len",  # as encoder
        "fc_name": "hidden2output",
        "fc_activation": None,
    },
]
# output
config_eval["model"].append(
    {
        "name": "output",
        "layers": [
            {
                "type": "output",
                "name": "output",
                "input": "RNN_section/outputs"
            }
        ],
    }
)
