import numpy as np
import tensorflow as tf
from autoencoder.model import Model
from config_RAE import config_eval as config

# Parameter
batch_size = 128
max_time_step = 10
data_size = 1

# Reconstruct Model
model = Model(config)

# Visualize Graph
writer = tf.summary.FileWriter("Log/RAE")
writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    # restore model
    #saver = tf.train.import_meta_graph("./Model/RAE/test_model-10000.meta")
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint("./Model/RAE"))
    
    graph = tf.get_default_graph()
    model_input = graph.get_tensor_by_name("inputs/input:0")
    model_output = graph.get_tensor_by_name("output/output:0")
    
    # Prepare Data
    d = np.linspace(0, max_time_step, max_time_step, endpoint=False).reshape(
        [1, max_time_step, data_size]
    )
    d = np.tile(d, (batch_size, 1, 1))
    r = np.random.randint(20, size=batch_size).reshape([batch_size, 1, 1])
    r = np.tile(r, (1, max_time_step, data_size))
    random_sequences = r + d
    rand_len = 10 - np.random.randint(3)
    random_sequences[:, rand_len:10, :] = -1

    # Testing
    (input_, output_) = sess.run([model_input, model_output], {model_input: random_sequences})
    print('train result:')
    print('input: ', input_[0, :, :].flatten())
    print('output: ', output_[0, :, :].flatten())

