# Autoencoder

Try to build many kinds of autoencoder. Currently got general autoencoder only.
VAE, RAE, and VRAE will be add in the future.

## Example
### AE
Use example from tensorflow example tutorial.
```
python example.py
```

## Usage
Use config to specify model architecture. See example in ```config.py```.
```python
from autoencoder.model import Model

# build model
model = Model(config)
# specity training parameters, not actually training
model.train(learning_rate)

# init tensorflow variables
sess.run(model.init)

# train
_, l = sess.run([model.optimizer, model.loss], feed_dict={model.input: batch_x})

# test full model
g = sess.run(model.decoder, feed_dict={model.input: batch_x})
# test encoder
g = sess.run(model.encoder, feed_dict={model.input: batch_x})

```

## Version
```tensorflow 1.7.1```
```tensorboard 1.7.0```

## Reference
[Tensorflow Example](https://github.com/aymericdamien/TensorFlow-Examples)
