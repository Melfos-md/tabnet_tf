import tensorflow as tf

from .attentive_transformer import AttentiveTransformer
from .feature_transformer import FeatureTransformer

class TabNetEncoder(tf.keras.Model):
    # Inputs
    # - previous_activation: activation from previous step a[i-1]
    # - features
    # Ouputs
    # - decisions d[i]
    # - activation: a[i]
    #
    def __init__(
            self,
            target_is_discrete, # true if target is a discrete variable
            N_step,
            num_features, # for attentive transformer
            output_units=1, # units of final dense layer: 1 for binary classification or regression, number of class for multiclass classification
            # Feature transformer hyperparameters
            N_a=8, 
            N_d=8,
            virtual_batch_size=256,
            momentum=0.99,
            epsilon=1e-5,
            relaxation_factor=1.0,
            # other
            seed=None
            ):
        super(TabNetEncoder, self).__init__()
        self.N_step = N_step
        self.target_is_discrete = target_is_discrete
        self.output_units = output_units
        self.num_features = num_features
        self.N_a = N_a
        self.N_d = N_d
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.relaxation_factor = relaxation_factor
        self.seed = seed
        
        self.input_feature_transformer = FeatureTransformer(N_a=self.N_a,
                                                      N_d=self.N_d,
                                                      shared=False,
                                                      virtual_batch_size=self.virtual_batch_size,
                                                      momentum=self.momentum,
                                                      epsilon=self.epsilon,
                                                      seed=self.seed)
        self.input_bn = tf.keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        self.tabnet_layers = [] # [[AttentiveTransformer, Step dependent FeatureTransformer] * N_step]
        for _ in range(self.N_step):
            attentive_transformer = AttentiveTransformer(num_features=self.num_features, relaxation_factor=self.relaxation_factor, 
                                                          seed=self.seed)
            feature_transformer = FeatureTransformer(N_a=self.N_a,
                                              N_d=self.N_d,
                                              shared=False,
                                              virtual_batch_size=self.virtual_batch_size,
                                              momentum=self.momentum,
                                              epsilon=self.epsilon,
                                              seed=self.seed)
            self.tabnet_layers.append([attentive_transformer, feature_transformer])

        self.relu = tf.keras.layers.ReLU()

        self.feature_transformer_shared = FeatureTransformer(N_a=self.N_a,
                                                      N_d=self.N_d,
                                                      shared=True,
                                                      virtual_batch_size=self.virtual_batch_size,
                                                      momentum=self.momentum,
                                                      epsilon=self.epsilon,
                                                      seed=self.seed)
        if target_is_discrete:
            self.dense_final = tf.keras.layers.Dense(units=output_units, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        else:
            self.dense_final = tf.keras.layers.Dense(units=1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))

    def call(self, inputs, training=None):
        features = self.input_bn(inputs, training=training)
        x = self.feature_transformer_shared(features, training=training)
        x = self.input_feature_transformer(x, training=training)
        
        activation, _ = tf.split(x, [self.N_a, self.N_d], axis=-1) # first decision is not used

        decision_out = 0
        agg_mask = 0
        
        for attentive_transformer, feature_transformer in self.tabnet_layers:
            mask = attentive_transformer(activation, training=training) # ith step AttentiveTransformer
            x = features * mask
            x = self.feature_transformer_shared(x, training=training)
            x = feature_transformer(x, training=training) # Decision step dependent FeatureTransformer
            activation, decision = tf.split(x, [self.N_a, self.N_d], axis=-1)
            decision = self.relu(decision)

            decision_out += decision # overall decision embedding (batch_size, N_d)

            # interpretability
            # compute eta_b[i]
            eta_b = tf.reduce_sum(decision, axis=1, keepdims=True)
            agg_mask += eta_b * mask # interpretability
        
        prediction = self.dense_final(decision_out)
        if self.target_is_discrete and not training:
            prediction = tf.argmax(prediction, axis=-1)

        if training:
            return agg_mask, prediction
        else:
            return prediction
        