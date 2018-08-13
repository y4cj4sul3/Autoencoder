# Autoencoder

Try to build many kinds of autoencoder. Currently got vanilla autoencoder and variantional autoencoder only.
RAE, and VRAE will be added in the future.

## Example

Use the example from tensorflow example tutorial.

### AE

The training process will save model in `Model/AE/` folder,
make sure the folder exist otherwise the files will failed to saved.<br>
During the evaluation, pretrained model will be restored for further testing.<br>

```
# train
python train_AE.py
# test
python eval_AE.py
```

Also, the model graph will be save in `Log/AE/` folder before training. Use the following command to visualize the model graph.

```
# launch tensorboard
tensorboard --logdir=Log/AE
```

The model configure defined in `config_AE.py`.

### VAE

Similar to AE, model saved in `Model/VAE/`, graph in `Log/VAE/`, and the model configure defined in `config_VAE.py`.

```
# train
python train_VAE.py
# test
python eval_VAE.py
# launch tensorboard
tensorboard --logdir=Log/VAE
```

Different to the previous example restores whole model for evaluation, this example only restores the decoder for using.<br>

## Usage

Use config to specify model architecture. See example in `config.py`.

```python
from autoencoder.model import Model

# build model
model = Model(config)
# specity training parameters, not actually training
model.train(learning_rate)

# init tensorflow variables
sess.run(model.init)

# train
sess.run(model.optimizer, feed_dict={model.input: data})

# test full model
sess.run(model.output, feed_dict={model.input: data})
# test encoder
sess.run(model.encoder, feed_dict={model.input: data})
# test decoder
sess.run(model.decoder, feed_dict={model.decoder_input: data})
```

## Version

`tensorflow 1.7.1`
`tensorboard 1.7.0`

## Reference

[Tensorflow Example](https://github.com/aymericdamien/TensorFlow-Examples)
