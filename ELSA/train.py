import tensorflow as tf
import numpy as np
import json
import sys
import os
from autoencoder import Model
import config_ELSA

# Arguments
ae_type = sys.argv[1]
cell_type = sys.argv[2]
hidden_size = int(sys.argv[3])
latent_size = int(sys.argv[4])
dir_path = sys.argv[5]
data_path = sys.argv[6]

# Dataset
with open(data_path, 'r') as fp:
  dataset = json.load(fp)

# Parameters
learning_rate = 0.001
iteration = 30000
batch_size = 100
max_seq_len = dataset['max_len']
data_size = len(dataset['data'][0][0])
batches = int(len(dataset['data']) / batch_size)

display_step = 100
save_step = 10000
decay_step = 2000

# Check Path
sub_path = dir_path+'/'+ae_type+'_'+cell_type+'_'+sys.argv[3]+'_'+sys.argv[4]
file_path = './Model/'+sub_path
if not os.path.exists(file_path):
  os.makedirs(file_path)

# Create Config
config = config_ELSA.createConfig(batch_size, max_seq_len, data_size, hidden_size, latent_size, ae_type, cell_type, "eval")

# Construct Model
model = Model(config)
model.train()

# Visualize Graph
writer = tf.summary.FileWriter('Log/'+sub_path)
writer.add_graph(tf.get_default_graph())

# Summary Log
sum_loss = tf.summary.scalar('loss', model.loss)
sum_rl = tf.summary.scalar('learning_rate', model.learning_rate)
sum_merged = tf.summary.merge_all()

# Prepare Data
data = [seq + [seq[-1]]*(max_seq_len-len(seq)) for seq in dataset['data']]
print(np.shape(data))

# Config
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth=True

# Start training
with tf.Session(config=config_tf) as sess:
  # Initialize
  sess.run(model.init)
  saver = tf.train.Saver()

  # get model input
  model_input = model.getNode('input')

  # Training
  for i in range(iteration):
    # Prepare Data
    if i % batches == 0:
      np.random.shuffle(data)
    
    batch_idx = (i % batches) * batch_size
    batch_x = data[batch_idx:batch_idx+batch_size]

    # Learning Rate Decay
    if i % decay_step == decay_step-1:
      learning_rate *= 0.1

    # Run Optimization
    _, l, s = sess.run(
      [model.optimizer, model.loss, sum_merged], feed_dict={model_input: batch_x, model.learning_rate: learning_rate}
    )

    # Display loss
    if i % display_step == display_step-1 or i == 0:
      print('Step %i, Loss: %f' % (i+1, l))
      writer.add_summary(s, i)
  
    # Save Model
    if i % save_step == save_step-1 and i+1 != iteration:
      saver.save(sess, file_path+'/test_model', global_step=i+1)
  
  # Save Model
  saver.save(sess, file_path+'/test_model', global_step=iteration)
