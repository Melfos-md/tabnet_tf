import tensorflow as tf

class Sparsemax(tf.keras.layers.Layer):
    """
    Sparsemax activation function.

    Arguments:
    logits -- tensor. Usually, (B, D) with B: batch size and D: number of features
    axis -- axis where sparsemax will be applied, default is -1 (so it is on features)

    Returns:
    Tensor of logits' shape
    """
    def __init__(self):
        super(Sparsemax, self).__init__()
    
    def call(self, logits):
        z = tf.sort(logits, axis=-1, direction='DESCENDING')
        k_values = 1 + tf.range(tf.shape(logits)[-1], dtype=logits.dtype)
        k_values = tf.expand_dims(k_values, axis=0)
        z_sum = tf.cumsum(z, axis=-1)
        comparison = z - (1.0 / k_values) * (z_sum - 1.0)

        mask= tf.cast(comparison > 0, logits.dtype)
        num_active = tf.reduce_sum(mask, axis=-1, keepdims=True)

        tau = (tf.reduce_sum(z * mask, axis=-1, keepdims=True) - 1.0) / num_active
        return tf.maximum(logits - tau, 0)
