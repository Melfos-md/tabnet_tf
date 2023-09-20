import tensorflow as tf

class GhostBatchNormalization(tf.keras.layers.Layer):
    """
    Implements the Ghost Batch Normalization technique.

    Reference: Hoffer, E.; Hubara, I.; and Soudry, D. 2017. Train longer,
    generalize better: closing the generalization gap in large batch
    training of neural networks. arXiv:1705.08741 .

    Ghost Batch Normalization (GBN) divides a larger batch into smaller virtual batches and 
    performs batch normalization on each smaller batch. This allows for larger overall batch 
    sizes during training while using batch normalization, which is usually sensitive to batch size.

    A design choice made here is to force user to choose a virtual batch size that divides the batch size without any remainder.

    Attributes:
    - virtual_batch_size (int): Size of each virtual batch.
    - momentum (float): Momentum for the moving average in batch normalization.
    - epsilon (float): Small constant added for numerical stability in batch normalization.
    - bn_layers (List[tf.keras.layers.BatchNormalization]): List of cached batch normalization layers.
    
    Methods:
    - call(inputs, training=None): Applies ghost batch normalization on the inputs.
    """
    def __init__(self, batch_size, virtual_batch_size=256, momentum=0.99, epsilon=1e-5):
        """
        Initialize the GhostBatchNormalization layer.

        Parameters:
        - batch_size (int): batch size.
        - virtual_batch_size (int): Size of each virtual batch. Default is 256.
        - momentum (float): Momentum for the moving average in batch normalization. Default is 0.99.
        - epsilon (float): Small constant added for numerical stability in batch normalization. Default is 1e-5.
        """
        super(GhostBatchNormalization, self).__init__()
        if batch_size % virtual_batch_size != 0:
            raise ValueError("batch_size must be a multiple of virtual_batch_size.")
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.bn_layers = [tf.keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon) 
                          for _ in range(batch_size // virtual_batch_size)]

    
    def call(self, inputs, training=None):
        """
        Applies ghost batch normalization on the inputs.

        Parameters:
        - inputs (tf.Tensor): Input tensor.
        - training (bool, optional): Whether the layer is in training mode. Default is None.

        Returns:
        - tf.Tensor: Output tensor after applying ghost batch normalization.
        """

        if tf.shape(inputs)[0] != self.batch_size:
            raise ValueError(f"'inputs' first dimension should be {self.batch_size}, but found {tf.shape(inputs)[0]}.")

        outputs = []
        for i, bn in enumerate(self.bn_layers):
            start = i * self.virtual_batch_size
            end = (i+1) * self.virtual_batch_size
            outputs.append(bn(inputs[start:end], training=training))
        return tf.concat(outputs, axis=0)