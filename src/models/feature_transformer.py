import tensorflow as tf
from .components.ghost_batchnormalization import GhostBatchNormalization
from .components.gated_linear_unit import GLU

class FeatureTransformer(tf.keras.layers.Layer):
    """
    Shared Feature Transformer shapes: (B, D) -> (B, N_a + N_d)
    """
    def __init__(self, N_a, N_d, virtual_batch_size, shared=True, momentum=0.99, epsilon=1e-5, seed=None, training=None):
        super(FeatureTransformer, self).__init__()
        self.N_a = N_a
        self.N_d = N_d
        self.N_fc = self.N_a + self.N_d
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.shared = shared
        self.seed = seed
        self.training = training

        self.fc1 = tf.keras.layers.Dense(units=self.N_fc, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.bn1 = GhostBatchNormalization(self.virtual_batch_size, self.momentum, self.epsilon)
        self.glu1 = GLU()

        self.fc2 = tf.keras.layers.Dense(units=self.N_fc, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.bn2 = GhostBatchNormalization(self.virtual_batch_size, self.momentum, self.epsilon)
        self.glu2 = GLU()


    def call(self, inputs):
        if inputs.shape[0] < self.virtual_batch_size:
            raise ValueError(f"Input batch size ({inputs.shape[0]}) should be greater than or equal to the virtual batch size ({self.virtual_batch_size}).")
        x = self.fc1(inputs)
        x = self.bn1(x, training=self.training)
        res = self.glu1(x)

        if not self.shared:
            if res.shape != inputs.shape:
                raise ValueError(f"Expected input shape {res.shape} found {inputs.shape}")
            else:
                res = (res + inputs) * tf.math.sqrt(0.5)

        x = self.fc2(res)
        x = self.bn2(x, training=self.training)
        x = self.glu2(x)

        x = (x + res) * tf.math.sqrt(0.5)
        return x

