import tensorflow as tf
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from autoencoder.model import Model

# from config_ELSA import config_eval as config
import config_ELSA


def l2norm(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2)))

def mse(a, b):
    return np.mean(np.mean(np.power(a-b, 2)))

# Arguments
ae_type = sys.argv[1]
cell_type = sys.argv[2]
hidden_size = int(sys.argv[3])
latent_size = int(sys.argv[4])
dir_path = sys.argv[5]
data_path = sys.argv[6]

# Dataset
with open(data_path, "r") as fp:
    dataset = json.load(fp)

# Parameters
batch_size = len(dataset["data"])
max_seq_len = dataset["max_len"]
data_size = len(dataset["data"][0][0])

# Reconstruct Model
config = config_ELSA.createConfig(
    batch_size,
    max_seq_len,
    data_size,
    hidden_size,
    latent_size,
    ae_type,
    cell_type,
    "eval",
)
model = Model(config)

# Visualize Graph
sub_path = dir_path + "/" + ae_type + "_" + cell_type + "_" + sys.argv[3] + "_" + sys.argv[4]
writer = tf.summary.FileWriter("Log/" + sub_path)
writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    # Restore Model
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint("./Model/" + sub_path))

    model_input = model.getNode("input")
    model_output = model.getNode("output")
    model_hidden = model.getNode("latent_code")

    # Prepare Data
    data = dataset["data"][0:batch_size]
    # data = sorted(data, key=lambda x: x[-1][2])
    # np.random.shuffle(data)
    print(np.shape(data))
    # padding
    data = [seq + [seq[-1]] * (max_seq_len - len(seq)) for seq in data]
    print(np.shape(data))

    # Testing
    _input, _hidden, _output, loss = sess.run(
        [model_input, model_hidden, model_output, model.loss], {model_input: data}
    )

    print("Loss: {}".format(loss))
    print("input:")
    print(_input[0, 0:10, :])
    print("output:")
    print(_output[0, 0:10, :])

    print(np.shape(_hidden))

    # Plot Trajectory

    # compare input output
    for i in range(max_seq_len-1):
      plt.plot(_input[0, i:i+2, 0], _input[0, i:i+2, 1], c=[0, 1, 0, 0.5])
      plt.plot(_output[0, i:i+2, 0], _output[0, i:i+2, 1], c=[1, 0, 0, float(i)/float(max_seq_len)])
      #plt.arrow(_output[0, i, 0], _output[0, i, 1], 0.1*_output[0, i, 2], 0.1*_output[0, i, 3])

    plt.axis([-1, 1, -1, 1])
    plt.show()

    # lstm
    if len(np.shape(_hidden)) == 3:
      _hidden = np.reshape(np.stack(_hidden, axis=1), (-1, 2*hidden_size))

    # compare latent code
    max_d = 0
    for i in range(batch_size):
      dist = mse(_hidden[0], _hidden[i])
      if dist > max_d:
        max_d = dist

    for i in range(batch_size):
      dist = mse(_hidden[0], _hidden[i])
      plt.plot(_output[i, :, 0], _output[i, :, 1], c=[np.sqrt(dist/max_d), 0, 0.5, 1])

    plt.plot(_input[0, :, 0], _input[0, :, 1], c=[1, 1, 0, 1])

    plt.axis([-1, 1, -1, 1])

    plt.show()

    '''
    for i in range(batch_size):
        # print('traj_0 vs traj_{}: traj_dist:{}, hidden_dist:{}'.format(i, l2norm(_input[0], _input[i]), l2norm(_hidden[0], _hidden[i]) ))
        print(
            "traj_0 vs traj_{}: hidden_dist:{}".format(
                i, l2norm(_hidden[0], _hidden[i])
            )
        )
    '''
