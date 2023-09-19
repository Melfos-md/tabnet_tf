import tensorflow as tf
import numpy as np
from models.tabnet_encoder import TabNetEncoder
from models.feature_transformer import FeatureTransformer

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def test_output_shape():
    # Hyperparameters
    batch_size = 1024
    N_a = N_d = 8
    num_features = 64
    output_units = 1

    features = tf.constant(np.random.rand(batch_size, num_features))

    agg_mask, decision_out = TabNetEncoder(target_is_discrete=False, output_units=output_units, N_step=4, num_features=num_features, N_a=N_a, N_d=N_d, seed=SEED)(features)

    assert agg_mask.shape.as_list() == [batch_size, num_features]
    assert decision_out.shape.as_list() == [batch_size, output_units]
