import tensorflow as tf
import numpy as np
import json
import sys
from autoencoder.model import Model
#from config_ELSA import config_eval as config
import config_ELSA

def l2norm(a, b):
  return np.sqrt(np.sum(np.power(a-b, 2)))

# Arguments
ae_type = sys.argv[1]
cell_type = sys.argv[2]
hidden_size = int(sys.argv[3])
latent_size = int(sys.argv[4])

# Dataset
with open('../raw_trajectory_3/testcase.json', 'r') as fp:
  dataset = json.load(fp)

# Parameters
batch_size = 10
max_seq_len = dataset['max_len']
data_size = len(dataset['data'][0][0])

# Reconstruct Model
config = config_ELSA.createConfig(batch_size, max_seq_len, data_size, hidden_size, latent_size, ae_type, cell_type, "eval")
model = Model(config)

# Visualize Graph
sub_path = 'ELSA/'+ae_type+'_'+cell_type+'_'+sys.argv[3]+'_'+sys.argv[4]
writer = tf.summary.FileWriter('Log/'+sub_path)
writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
  # Restore Model
  saver = tf.train.Saver()
  saver.restore(sess, tf.train.latest_checkpoint('./Model/'+sub_path))

  model_input = model.getNode('input')
  model_output = model.getNode('output')
  model_hidden = model.getNode('latent_code')

  # Prepare Data
  data = dataset['data'][0:batch_size]
  data = sorted(data, key=lambda x: x[-1][0])
  print(np.shape(data))
  # padding
  data = [seq + [[0]*data_size]*(max_seq_len-len(seq)) for seq in data]
  print(np.shape(data))

  # Testing
  _input, _hidden, _output, loss = sess.run([model_input, model_hidden, model_output, model.loss], {model_input: data})

  print('Loss: {}'.format(loss))
  print('input:')
  print(_input[0, 0:10, :])
  print('output:')
  print(_output[0, 0:10, :])

  print(np.shape(_hidden))

  for i in range(batch_size):
    print('{} vs {}: {}'.format(dataset['data'][0][-1][0], dataset['data'][i][-1][0], l2norm(_hidden[0], _hidden[i]) ))


