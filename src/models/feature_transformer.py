import tensorflow as tf
from .components.ghost_batchnormalization import GhostBatchNormalization
from .components.gated_linear_unit import GLU

class FeatureTransformer(tf.keras.layers.Layer):
    """
    FeatureTransformer Layer for TabNet.
    
    This layer serves as a key building block in the TabNet architecture. It processes the input features through
    two successive transformations, providing both standard and ghost batch normalization capabilities. Designed 
    to extract and refine the features from input data, the layer prepares the transformed features for subsequent 
    stages in the TabNet model.

    Attributes:
    - N_a (int): Dimension of the output specific to attention transformations.
    - N_d (int): Dimension of the output specific to decision transformations.
    - virtual_batch_size (int): Size of the batches on which the Ghost Batch Normalization operates.
    - shared (bool, default=True): If set to True, the architecture will correspond to the architecture shared across decision steps; 
        if set to False, it will adopt a decision-step-dependent architecture, as described in the paper.
    - momentum (float, default=0.99): Momentum for the moving average in Ghost Batch Normalization.
    - epsilon (float, default=1e-5): Small constant to avoid division by zero in Ghost Batch Normalization.
    - seed (int, optional): Random seed for initializers.
    - training (bool, optional): Training mode flag for Ghost Batch Normalization.

    Methods:
    - `call(inputs)`
        - `inputs` (Tensor): Input tensor of shape `(batch_size, num_features)`

    Returns: A tensor of shape `(batch_size, N_a + N_d)`
    During the `call` method, the layer checks if the input batch size is larger than or equal to the 
    virtual batch size. This check ensures the correct operation of the Ghost Batch Normalization. 
    It then processes the input through two transformation blocks, consisting of a Dense layer, 
    Ghost Batch Normalization, and GLU activation. If the `shared` attribute is set to False, 
    it combines the output from the first transformation with the original input using an element-wise 
    addition. This combined output is then scaled down by a factor of `sqrt(0.5)` (See note on `sqrt(0.5)` in README.md).
    """
    def __init__(self, N_a, N_d, shared, virtual_batch_size=256, momentum=0.99, epsilon=1e-5, seed=None):
        super(FeatureTransformer, self).__init__()
        self.N_a = N_a
        self.N_d = N_d
        self.N_fc = self.N_a + self.N_d
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.shared = shared
        self.seed = seed

        self.fc1 = tf.keras.layers.Dense(units=self.N_fc, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.bn1 = GhostBatchNormalization(self.virtual_batch_size, self.momentum, self.epsilon)
        self.glu1 = GLU()

        self.fc2 = tf.keras.layers.Dense(units=self.N_fc, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.bn2 = GhostBatchNormalization(self.virtual_batch_size, self.momentum, self.epsilon)
        self.glu2 = GLU()


    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.bn1(x, training=training)
        res = self.glu1(x)

        if not self.shared:
            if res.shape != inputs.shape:
                raise ValueError(f"Expected input shape {res.shape} found {inputs.shape}")
            else:
                res = (res + inputs) * tf.math.sqrt(0.5)

        x = self.fc2(res)
        x = self.bn2(x, training=training)
        x = self.glu2(x)

        x = (x + res) * tf.math.sqrt(0.5)
        return x

