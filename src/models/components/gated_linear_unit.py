import tensorflow as tf

class GLU(tf.keras.layers.Layer):
    def __init__(self):
        """Gated Linear Unit (GLU) layer."""
        super(GLU, self).__init__()
    
    def call(self, inputs):
        """
        Apply GLU activation.
        
        Args:
            inputs (tf.Tensor): Output of the Batch Normalization layer (post FC).
            
        Returns:
            tf.Tensor: Output after applying GLU.
        """
        # Ensure the input is of the expected shape
        if len(inputs.shape) != 2:
            raise ValueError("Expected inputs to be a 2D tensor")
        
        # Compute the GLU activation
        gate = tf.sigmoid(inputs)
        return inputs * gate