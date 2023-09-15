import tensorflow as tf

class FeatureTransformerShared(tf.keras.layers.Layer):
    def __init__(self, N_a, N_d):
        super(FeatureTransformerShared, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=N_a)

    def call(self, inputs):
        self.fc1 = tf.keras.layers.Dense()