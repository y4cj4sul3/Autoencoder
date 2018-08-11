import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder.model import Model
from config import config

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Construct model
model = Model(config)
model.train(learning_rate)

# Visualize Graph
writer = tf.summary.FileWriter("Log")
writer.add_graph(tf.get_default_graph())

# Start
with tf.Session() as sess:
    # Initialize
    sess.run(model.init)

    # Traning
    for i in range(1, num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run Optimization
        _, l = sess.run([model.optimizer, model.loss], feed_dict={model.input: batch_x})
        # Display loss
        if i % display_step == 0 or i == 1:
            print("Step %i: Minibatch Loss: %f" % (i, l))

    # Testing
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(model.decoder, feed_dict={model.input: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = batch_x[
                j
            ].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = g[j].reshape(
                [28, 28]
            )

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
