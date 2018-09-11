import tensorflow as tf
import copy
'''
# Network Parameters
batch_size = 10
data_size = 6
max_time_step = 172
num_hidden = 256
'''
# custom function
def teacher_forcing_input(inputs):
    # input shape should be [batch_size, time_step, data_size]
    #_input = inputs[0]
    # get zero padding shape
    pad_shape = inputs.get_shape().as_list()
    pad_shape[1] = 1
    # pad at t=0 and remove t=T
    return tf.concat([tf.zeros(pad_shape), inputs[:, :-1]], axis=1)

# create config
def createConfig(batch_size, max_time_step, data_size, hidden_size, latent_size, ae_type='RAE', cell_type='LSTM', mode="train"):
    config = {
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

        ],
        "loss": [
            {
                "name": "enc_dec_loss",
                "weight": 1,
                "ground_truth": "input",
                "prediction": "output",
            }
        ]

    }

    # cell
    if cell_type == 'LSTM':
        cell = tf.contrib.rnn.BasicLSTMCell
    elif cell_type == 'GRU':
        cell = tf.contrib.rnn.GRUCell
    else:
        cell = tf.contrib.rnn.BasicRNNCell

    # encoder
    config['model'].append(
        {
            "name": "encoder",
            "layers": [
                {
                    "type": "RNN",
                    "name": "RNN_enc",
                    "input": "input",
                    "input_mode": "INPUT_MODE",
                    "init_state": None,
                    "cell": cell,
                    "output_size": hidden_size,
                    "activation": None,
                    "sequence_len": max_time_step,
                },
            ]
        }
    )
    if ae_type == 'VRAE':
        config['model'][1]['layers'].append(
            {
                "type": "sampler",
                "name": "sampler",
                "input": "RNN_enc/state",
                "output_size": latent_size,
            },
        )
        config['model'][1]['layers'].append(
            {
                "type": "output",
                "name": "latent_code",
                "input": "sampler",
            }
        )
    else:
        config['model'][1]['layers'].append(
            {
                "type": "output",
                "name": "latent_code",
                "input": "RNN_enc/state",
            }
        )

    # decoder
    if mode == 'train':
        # teacher forcing
        config['model'].append(
            {
                "name": "decoder",
                "layers": [
                    {
                        "type": "custom_function",
                        "name": "teacher_forcing_input",
                        "input": "input",
                        "function": teacher_forcing_input
                    },
                ]
            }
        )
        if ae_type == "VRAE":
            # latent code to hidden state
            config['model'][2]['layers'].append(
                {
                    "type": "FC",
                    "name": "latent2hidden",
                    "input": "latent_code",
                    "output_size": hidden_size,
                    "activation": tf.nn.tanh,
                }
            )
            config['model'][2]['layers'].append(
                {
                    "type": "RNN",
                    "name": "RNN_dec",
                    "input": "teacher_forcing_input",
                    "input_mode": "INPUT_MODE",
                    "init_state": "latent2hidden",
                    "cell": cell,
                    "output_size": hidden_size,
                    "activation": None,
                    "sequence_len": max_time_step,  # as encoder
                    "fc_activation": None,
                },
            )
        else:
            # RAE
            config['model'][2]['layers'].append(
                {
                    "type": "RNN",
                    "name": "RNN_dec",
                    "input": "teacher_forcing_input",
                    "input_mode": "INPUT_MODE",
                    "init_state": "latent_code",
                    "cell": cell,
                    "output_size": hidden_size,
                    "activation": None,
                    "sequence_len": max_time_step,  # as encoder
                    "fc_activation": None,
                },
            )
        # hidden state to output
        config['model'][2]['layers'].append(
            {
                "type": "reshape",
                "name": "flatten",
                "input": "RNN_dec/outputs",
                "output_size": [-1, hidden_size],
            },
        )
        config['model'][2]['layers'].append(
            {
                "type": "FC",
                "name": "hidden2output",
                "input": "flatten",
                "output_size": data_size,
                "activation": None
            },
        )
        config['model'][2]['layers'].append(
            {
                "type": "reshape",
                "name": "output",
                "input": "hidden2output",
                "output_size": [batch_size, max_time_step, data_size]
            }
        )

    elif mode == "eval":
        if ae_type == "VRAE":
            config['model'].append(
                {
                    "name": "decoder",
                    "layers": [
                        {
                            "type": "FC",
                            "name": "latent2hidden",
                            "input": "latent_code",
                            "output_size": hidden_size,
                            "activation": tf.nn.tanh,
                        },
                        {
                            "type": "RNN",
                            "name": "RNN_dec",
                            "input": "RNN_enc/input_size",  # specify data size
                            "input_mode": "OUTPUT_MODE",
                            "init_state": "latent2hidden",
                            "cell": cell,
                            "output_size": hidden_size,
                            "activation": None,
                            "sequence_len": "RNN_enc/sequence_len",  # as encoder
                            "fc_name": "hidden2output",
                            "fc_activation": None,
                        },
                    ]
                }
            )
        else:
            config['model'].append(
                {
                    "name": "decoder",
                    "layers": [
                        {
                            "type": "RNN",
                            "name": "RNN_dec",
                            "input": "RNN_enc/input_size",  # specify data size
                            "input_mode": "OUTPUT_MODE",
                            "init_state": "latent_code",
                            "cell": cell,
                            "output_size": hidden_size,
                            "activation": None,
                            "sequence_len": "RNN_enc/sequence_len",  # as encoder
                            "fc_name": "hidden2output",
                            "fc_activation": None,
                        },
                    ]
                }
            )
        config["model"].append(
            {
                "name": "output",
                "layers": [
                    {
                        "type": "output",
                        "name": "output",
                        "input": "RNN_dec/outputs"
                    }
                ],
            }
        )

    return config


