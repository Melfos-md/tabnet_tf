import tensorflow as tf

class GhostBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, virtual_batch_size, momentum=0.99, epsilon=1e-5):
        super(GhostBatchNormalization, self).__init__()
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.bn_layers = []

    def build(self, input_shape):
        # Create a list of individual batch normalization layers for each ghost batch
        for _ in range(input_shape[0] // self.virtual_batch_size):
            self.bn_layers\
                .append(
                    tf.keras.layers.BatchNormalization(momentum=self.momentum, 
                                                       epsilon=self.epsilon)
                    )
    
    def call(self, inputs, training=None):
        outputs = []
        # Split inputs into ghost batches and apply individual BN
        for i, bn in enumerate(self.bn_layers):
            start = i * self.virtual_batch_size
            end = (i+1) * self.virtual_batch_size
            outputs.append(bn(inputs[start:end], training=training))
        # Concatenate the outputs to obtain the original batch size
        return tf.concat(outputs, axis=0)