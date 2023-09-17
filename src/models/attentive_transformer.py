# Implementation of attentive transformer

import tensorflow as tf
from models.components.sparsemax import Sparsemax


class AttentiveTransformer(tf.keras.layers.Layer):
  """
  Attentive Transformer as introduced in the TabNet architecture.
  
  The transformer applies a combination of linear transformation, 
  batch normalization, and sparsemax activation to the input features 
  and scales the result by a prior. It returns a mask which highlights 
  certain features over others. The prior is updated based on the mask 
  and a relaxation factor.
  
  Arguments:
  - seed (int, optional): Random seed for initialization. Used 
                          for unit tests. Defaults to None.
  - relaxation_factor (float): Value by which to scale the updated prior. 
                               Defaults to 1.0.
                               
  Input shape:
  - (batch_size, num_features)
  Output shape:
  - (batch_size, num_features)
  Example:
  ```python
  transformer = AttentiveTransformer()
  output_mask = transformer(input_tensor)
  ```


  Attributes:
  - batch_norm (layers.BatchNormalization): Batch normalization layer.
  - sparsemax (Sparsemax): Custom Sparsemax activation function.
  - seed (int or None): Random seed for the dense layer initialization.
  - relaxation_factor (float): Value to scale the updated prior.
  - prior_scales (tf.Variable or None): Updated prior values based on previous mask.

  References:
  - [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
  """

  def __init__(self, num_features, relaxation_factor=1.0, seed=None):
    super(AttentiveTransformer, self).__init__()
    self.num_features = num_features
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.sparsemax = Sparsemax()
    self.seed = seed # for unit tests
    self.relaxation_factor = relaxation_factor
    self.prior_scales = None

  def build(self, input_shape):
    """
    Create the layer's weights.

    Raises:
        ValueError: if input shape is not of rank 2.
    """
    if len(input_shape) != 2:
      raise ValueError("Expected input shape (batch_size, num_features), but received a different shape")

    self.fc = tf.keras.layers.Dense(units=self.num_features, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))

  def call(self, inputs, training=None):

    """
    Compute the Attentive Transformer's output mask.

    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, num_features).

    Returns:
        tf.Tensor: Output mask of shape (batch_size, num_features).
    """
    # Initialize prior_scales if it's the first call
    if self.prior_scales is None:
      initial_value = tf.ones((inputs.shape[0], self.num_features)) # (batch_size, num_features) same as features and mask
      self.prior_scales = tf.Variable(initial_value, trainable=False, dtype=tf.float32)

    x = self.fc(inputs) 
    x = self.batch_norm(x, training=training)
    x = tf.multiply(x, self.prior_scales)
    mask = self.sparsemax(x)

    # Update prior_scales
    self.prior_scales.assign(self.prior_scales * (self.relaxation_factor - mask))
    return mask # must be (batch_size, num_features)

