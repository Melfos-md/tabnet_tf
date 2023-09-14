import tensorflow as tf

class FeatureTransformer(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureTransformer, self).__init__()

    def call(self, inputs):