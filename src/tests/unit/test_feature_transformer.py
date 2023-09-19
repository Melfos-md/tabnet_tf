import tensorflow as tf
import numpy as np
from models.feature_transformer import FeatureTransformer

SEED = 42
np.random.seed(42)
tf.random.set_seed(42)

def test_shared_transformer_output_shape():
    BATCH_SIZE = 1024
    NUM_FEATURES = 2048

    N_a = 16
    N_d = 8

    input = tf.constant(np.random.rand(BATCH_SIZE, NUM_FEATURES)) # input shape (B, D)
    output = FeatureTransformer(N_a, N_d, shared=True, virtual_batch_size=32, seed=SEED)(input)

    assert output.shape.as_list() == [BATCH_SIZE, N_d+N_a]

def test_not_shared_transformer_output_shape():
    BATCH_SIZE = 1024

    N_a = 16
    N_d = 8

    input = tf.constant(np.random.rand(BATCH_SIZE, N_a + N_d)) # input shape (B, N_a+N_d)
    output = FeatureTransformer(N_a, N_d, virtual_batch_size=32, shared=False, seed=SEED)(input)

    assert output.shape.as_list() == [BATCH_SIZE, N_a+N_d]

