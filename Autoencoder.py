import tensorflow as tf
import numpy as np
import json
from autoencoder.model import Model
import config_ELSA

class Autoencoder:
  '''
  load pretrained autoencoder
  '''
  def __init__(self, sess, ae_type, cell_type, hidden_size, latent_size, max_seq_len, data_size):
    '''
    init model
    ```sess```: tensorflow session
    ```ae_type```: 'RAE' or 'VRAE'
    ```cell_type```: 'LSTM', 'GRU', or 'RNN'
    ```hidden_size```: RNN cell hidden units
    ```latent_size```: latent code size
    ```max_seq_len```: recurrent count
    ```data_size```: data size per step
    '''
    self.sess = sess

    # Parameters
    self.batch_size = 1
    self.max_seq_len = max_seq_len
    self.data_size = data_size

    # Model Type
    self.ae_type = ae_type
    self.cell_type = cell_type
    self.hidden_size = hidden_size
    self.latent_size = latent_size

    # Reconstruct Model
    config = config_ELSA.createConfig(
      self.batch_size,
      self.max_seq_len,
      self.data_size,
      self.hidden_size,
      self.latent_size,
      self.ae_type,
      self.cell_type,
      'eval'
    )
    self.model = Model(config)

    # Restore Model
    #print(cell_type)
    #print(self.latent_size)
    #print('{}_{}'.format(self.hidden_size, self.latent_size))
    file_path = 'Model/ELSA/'+self.ae_type+'_'+self.cell_type+'_{}_{}'.format(self.hidden_size, self.latent_size)
    saver = tf.train.Saver()
    saver.restore(self.sess, tf.train.latest_checkpoint(file_path))

    # Retrieve Nodes
    self.model_input = self.model.getNode('input')
    self.model_output = self.model.getNode('output')
    self.model_latent = self.model.getNode('latent_code')

  def encode(self, input_data):
    '''
    ```input_data``` should have shape [1, max_seq_len, data_size]
    '''

    _encoded = self.sess.run(self.model_latent, {self.model_input: input_data})
    return _encoded



