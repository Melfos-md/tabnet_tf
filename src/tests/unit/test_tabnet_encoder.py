import tensorflow as tf
import numpy as np
from models.tabnet_encoder import TabNetEncoder

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def test_output_shape():

    # Hyperparameters
    batch_size = 1024
    N_a = N_d = 8
    num_features = 64
    output_units = 1

    # Generate random input features
    features = tf.constant(np.random.rand(batch_size, num_features))

    # Initialize and call the TabNetEncoder model
    model = TabNetEncoder(target_is_discrete=False, output_units=output_units, N_step=4, num_features=num_features, N_a=N_a, N_d=N_d, seed=SEED)
    agg_mask, decision_out = model(features, training=True)

    # Check if the output shapes match the expected shapes
    assert agg_mask.shape.as_list() == [batch_size, num_features]
    assert decision_out.shape.as_list() == [batch_size, output_units]
