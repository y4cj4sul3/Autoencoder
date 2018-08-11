import tensorflow as tf

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_input = 784  # MNIST data input (img shape: 28*28)

# Model Architecture Configure
config = {
    "input_shape": [None, num_input],
    "encoder": [
        {
            "type": "FC",
            "input_size": num_input,
            "output_size": num_hidden_1,
            "activation": tf.nn.sigmoid,
        },
        {
            "type": "FC",
            "input_size": num_hidden_1,
            "output_size": num_hidden_2,
            "activation": tf.nn.sigmoid,
        },
    ],
    "sampler": None,
    "decoder": [
        {
            "type": "FC",
            "input_size": num_hidden_2,
            "output_size": num_hidden_1,
            "activation": tf.nn.sigmoid,
        },
        {
            "type": "FC",
            "input_size": num_hidden_1,
            "output_size": num_input,
            "activation": tf.nn.sigmoid,
        },
    ],
}
