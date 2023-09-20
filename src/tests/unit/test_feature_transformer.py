import tensorflow as tf
import numpy as np
import pytest
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
    output = FeatureTransformer(N_a, N_d, shared=True, batch_size=BATCH_SIZE, virtual_batch_size=32, seed=SEED)(input)

    assert output.shape.as_list() == [BATCH_SIZE, N_d+N_a]

def test_not_shared_transformer_output_shape():
    BATCH_SIZE = 1024

    N_a = 16
    N_d = 8

    input = tf.constant(np.random.rand(BATCH_SIZE, N_a + N_d)) # input shape (B, N_a+N_d)
    output = FeatureTransformer(N_a, N_d, batch_size=BATCH_SIZE, virtual_batch_size=32, shared=False, seed=SEED)(input)

    assert output.shape.as_list() == [BATCH_SIZE, N_a+N_d]

def test_feature_transformer_batch_size_check():
    num_features = 10

    N_a = 8
    N_d = 8
    virtual_batch_size = 32

    input_tensor = tf.random.normal((virtual_batch_size - 1, num_features))
    # Check that the ValueError is raised when the input batch size is smaller than the virtual batch size
    with pytest.raises(ValueError):
        FeatureTransformer(N_a=N_a, N_d=N_d, batch_size=virtual_batch_size - 1, shared=True, virtual_batch_size=virtual_batch_size)(input_tensor)

    input_tensor = tf.random.normal((virtual_batch_size, num_features))

    # Check that no error is raised when the input batch size is equal to the virtual batch size
    FeatureTransformer(N_a=N_a, N_d=N_d, batch_size=virtual_batch_size, shared=True, virtual_batch_size=virtual_batch_size)(input_tensor)

    # Create a dummy input tensor with a batch size larger than the virtual batch size
    batch_size = virtual_batch_size + 32
    input_tensor = tf.random.normal((batch_size, num_features))

    # Check that no error is raised when the input batch size is larger than the virtual batch size
    FeatureTransformer(N_a=N_a, N_d=N_d, batch_size=batch_size, shared=True, virtual_batch_size=virtual_batch_size)(input_tensor)