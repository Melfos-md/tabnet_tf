# Implementation of attentive transformer

import tensorflow as tf
from tensorflow import keras
from keras import layers


'''
From the paper, I understand that FC means only the linear part so there is no activation in the attentive transformer
'''
# Hyperparameters: units (FC number of units)
class AttentiveTransformer(layers.Layer):
  def __init__(self, units):
    super(AttentiveTransformer, self).__init__()
    self.fc = layers.Dense(units, activation=None)
    self.batch_norm = layers.BatchNormalization()

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

