import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

import Autoencoder as ae


def mse(a, b):
  return np.mean(np.power(a-b, 2))

with open('./Model/ELSA_2/dataset/testing_2.json', 'r') as fp:
  data = json.load(fp)

with tf.Session() as sess:
  # parameters
  seq_len = data['max_len']
  data_size = len(data['data'][0][0])
  batch = len(data['data'])
  
  # construct model
  model = ae.Autoencoder(sess, 'Model/ELSA_2/', 'RAE', 'GRU', 128, 128, 1, seq_len, data_size)

  # padding
  data = np.array([seq + [seq[-1]]*(seq_len-len(seq)) for seq in data['data']])
  print(np.shape(data))
  
  hidden_0 = model.encode([data[0]])

  # compare latent code
  max_d = 0
  dists = []
  for i in range(batch):
    hid = model.encode([data[i]])
    dist = mse(hidden_0, hid)
    if dist > max_d:
      max_d = dist
    dists.append(dist)

  for i in range(batch):
    plt.plot(data[i, :, 0], data[i, :, 1], c=[np.sqrt(dists[i]/max_d), 0, 0.5, 1])

  plt.plot(data[0, :, 0], data[0, :, 1], c=[1, 1, 0, 1])

  plt.axis([-1, 1, -1, 1])

  plt.show()
