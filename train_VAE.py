from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder.model import Model
from config_VAE import config

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
num_steps = 30000
batch_size = 64

display_step = 1000

# Construct model
model = Model(config)
model.train(learning_rate)

# Visualize Graph
writer = tf.summary.FileWriter("Log/VAE")
writer.add_graph(tf.get_default_graph())

# Start training
with tf.Session() as sess:
    # Initialize
    sess.run(model.init)

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run Optimization
        _, l, edl, kll = sess.run(
            [model.optimizer, model.loss, model.encode_decode_loss, model.kl_div_loss],
            feed_dict={model.input: batch_x},
        )
        # Display loss
        if i % display_step == 0 or i == 1:
            print(
                "Step %i, Loss: %f, EDL: %f, KLL: %f" % (i, l, np.sum(edl), np.sum(kll))
            )

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, "./Model/VAE/test_model", global_step=num_steps)
