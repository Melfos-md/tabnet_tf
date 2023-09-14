# Implementation of attentive transformer

import tensorflow as tf
from tensorflow import keras
from keras import layers
from models.sparsemax import Sparsemax


class AttentiveTransformer(layers.Layer):
  """
  Implement Attentive Transformer.
  Arguments:
  input_shape: must be (batch_size, num_features) shape

  Return
  Attentive Transformer layer.

  The output shape of this layer is (1, num_features).

  """
  def __init__(self, seed=None):
    super(AttentiveTransformer, self).__init__()
    self.batch_norm = layers.BatchNormalization()
    self.sparsemax = Sparsemax()
    self.seed = seed # for unit tests

  def build(self, input_shape):
    if len(input_shape) != 2:
      raise ValueError("Expected input shape (batch_size, num_features), but received a different shape")
    batch_size = input_shape[0]
    num_features = input_shape[-1]
    self.prior_scales = self.add_weight("prior_scales", 
                                  shape=(batch_size, num_features), 
                                  initializer="ones", 
                                  trainable=True)
    self.fc = layers.Dense(units=num_features, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))

  def call(self, inputs):
    x = self.fc(inputs)
    x = self.batch_norm(x)
    x = tf.multiply(x, self.prior_scales)
    return self.sparsemax(x)

