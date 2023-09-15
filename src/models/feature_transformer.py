import tensorflow as tf

class FeatureTransformerShared(tf.keras.layers.Layer):
    def __init__(self, N_a, N_d):
        super(FeatureTransformerShared, self).__init__()
        self.N_a = N_a
        self.N_d = N_d
        self.N_fc = self.N_a + self.N_d
        self.fc1 = tf.keras.layers.Dense(units=self.N_fc, activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(units=self.N_fc)

    def call(self, inputs):
