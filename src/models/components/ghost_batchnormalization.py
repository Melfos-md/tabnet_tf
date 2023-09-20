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

    A design choice made here is to create or fetch batch normalization layers on-the-fly. 
    The reason behind this is the need to adapt to the actual batch size during the `call` method, 
    which might vary during training. By dynamically ensuring that the right number of 
    batch normalization layers are present, we cater to different actual batch sizes that might 
    be encountered.

    Attributes:
    - virtual_batch_size (int): Size of each virtual batch.
    - momentum (float): Momentum for the moving average in batch normalization.
    - epsilon (float): Small constant added for numerical stability in batch normalization.
    - bn_layers (List[tf.keras.layers.BatchNormalization]): List of cached batch normalization layers.
    
    Methods:
    - _get_or_create_bn(index): Retrieves (or creates and caches if not existing) a batch normalization layer
                                based on the given index.
    - call(inputs, training=None): Applies ghost batch normalization on the inputs.
    """
    def __init__(self, virtual_batch_size=256, momentum=0.99, epsilon=1e-5):
        """
        Initialize the GhostBatchNormalization layer.

        Parameters:
        - virtual_batch_size (int): Size of each virtual batch. Default is 256.
        - momentum (float): Momentum for the moving average in batch normalization. Default is 0.99.
        - epsilon (float): Small constant added for numerical stability in batch normalization. Default is 1e-5.
        """
        super(GhostBatchNormalization, self).__init__()
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.bn_layers = []

    def _get_or_create_bn(self, index):
        """
        Retrieves a batch normalization layer from cache based on the given index.
        If the layer doesn't exist, creates a new one, caches it, and returns it.

        Parameters:
        - index (int): Index for the batch normalization layer.

        Returns:
        - tf.keras.layers.BatchNormalization: Batch normalization layer.
        """
        if index < len(self.bn_layers):
            return self.bn_layers[index]
        bn = tf.keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        self.bn_layers.append(bn)
        return bn
    
    def call(self, inputs, training=None):
        """
        Applies ghost batch normalization on the inputs.

        Parameters:
        - inputs (tf.Tensor): Input tensor.
        - training (bool, optional): Whether the layer is in training mode. Default is None.

        Returns:
        - tf.Tensor: Output tensor after applying ghost batch normalization.
        """
        outputs = []
        actual_batch_size = tf.shape(inputs)[0]
        num_ghost_batches = actual_batch_size // self.virtual_batch_size
        
        # Use cached or create new BN layers for each ghost batch
        for i in range(num_ghost_batches):
            bn = self._get_or_create_bn(i)
            start = i * self.virtual_batch_size
            end = (i+1) * self.virtual_batch_size
            outputs.append(bn(inputs[start:end], training=training))

        if actual_batch_size % self.virtual_batch_size != 0:
            bn = self._get_or_create_bn(i)
            outputs.append(bn(inputs[num_ghost_batches * self.virtual_batch_size:], training=training))
        # Concatenate the outputs to obtain the original batch size
        return tf.concat(outputs, axis=0)