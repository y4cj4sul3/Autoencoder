import tensorflow as tf
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from autoencoder import Autoencoder as ae

def mse(a, b):
  return np.mean(np.power(a-b, 2))

# Arguments
ae_type = sys.argv[1]
cell_type = sys.argv[2]
hidden_size = int(sys.argv[3])
latent_size = int(sys.argv[4])
dir_path = sys.argv[5]
data_path = sys.argv[6]

# Parameters
with open(data_path, 'r') as fp:
  data = json.load(fp)

with tf.Session() as sess:
  # parameters
  seq_len = data['max_len']
  data_size = len(data['data'][0][0])
  batch = len(data['data'])
  
  # construct model
  model = ae.Autoencoder(sess, 'Model/'+dir_path+'/', ae_type, cell_type, hidden_size, latent_size, batch, seq_len, data_size)

  # padding
  data = np.array([seq + [seq[-1]]*(seq_len-len(seq)) for seq in data['data']])
  print(np.shape(data))
  
  hiddens = model.encode(data)

  # compare latent code
  max_d = 0
  dists = []
  for i in range(batch):
    dist = mse(hiddens[0], hiddens[i])
    if dist > max_d:
      max_d = dist
    dists.append(dist)

  for i in range(batch):
    plt.plot(data[i, :, 0], data[i, :, 1], c=[(dists[i]/max_d), 0, 0.5, 1])

  plt.plot(data[0, :, 0], data[0, :, 1], c=[1, 1, 0, 1])

  plt.axis([-10, 10, -10, 10])

  plt.show()
