import tensorflow as tf

class Sparsemax(tf.keras.layers.Layer):
    """
    Implementation of the Sparsemax activation function as introduced by 
    André F. T. Martins and Ramón Fernandez Astudillo.
    
    The Sparsemax activation transforms the input tensor by projecting it onto the 
    simplex, producing a sparse output. It's particularly useful as an alternative 
    to the softmax function where a more sparse representation is desired.
    
    Arguments:
    - None.

    Input shape:
    - 2D tensor: (batch_size, num_features).
    
    Output shape:
    - Same shape as the input.

    Example:
    ```python
    layer = Sparsemax()
    output = layer(input_tensor)
    ```

    References:
    - [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068)

    """
    def __init__(self):
        super(Sparsemax, self).__init__()
    
    def call(self, logits):
        """
        Compute the Sparsemax activation value.

        Args:
            logits (tf.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            tf.Tensor: Transformed tensor with the same shape as the input.
        """
        z = tf.sort(logits, axis=-1, direction='DESCENDING')
        k_values = 1 + tf.range(tf.shape(logits)[-1], dtype=logits.dtype)
        k_values = tf.expand_dims(k_values, axis=0)
        z_sum = tf.cumsum(z, axis=-1)
        comparison = z - (1.0 / k_values) * (z_sum - 1.0)

        mask= tf.cast(comparison > 0, logits.dtype)
        num_active = tf.reduce_sum(mask, axis=-1, keepdims=True)

        tau = (tf.reduce_sum(z * mask, axis=-1, keepdims=True) - 1.0) / num_active
        return tf.maximum(logits - tau, 0)
