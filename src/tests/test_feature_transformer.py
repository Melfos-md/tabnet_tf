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
    output = FeatureTransformer(N_a, N_d, virtual_batch_size=32, seed=SEED)(input)

    assert output.shape.as_list() == [BATCH_SIZE, N_d+N_a]

def test_not_shared_transformer_output_shape():
    BATCH_SIZE = 1024

    N_a = 16
    N_d = 8

    input = tf.constant(np.random.rand(BATCH_SIZE, N_a + N_d)) # input shape (B, N_a+N_d)
    output = FeatureTransformer(N_a, N_d, virtual_batch_size=32, shared=False, seed=SEED)(input)

    assert output.shape.as_list() == [BATCH_SIZE, N_a+N_d]

def test_random_value():
    N_a = N_d = 2

    input = tf.constant(np.random.rand(5, 10))
    output = FeatureTransformer(N_a=N_a, N_d=N_d, virtual_batch_size=1, shared=True, seed=SEED)(input)
    output = FeatureTransformer(N_a=N_a, N_d=N_d, virtual_batch_size=1, shared=False, seed=SEED)(output)
    d, a = tf.split(output, [N_a, N_d], axis=-1)
    print(d)

    expected_a = np.array([[ 0.04515576,  0.01253379],
                            [ 0.2286984,  -0.10788213],
                            [ 0.4388903,  -0.1324857 ],
                            [ 0.15477486, -0.10506041],
                            [ 0.23625115, -0.08616962]])
    expected_d = np.array([[-0.06584572, -0.07190628],
                            [-0.12262455, -0.19794567],
                            [-0.17495504, -0.3395235 ],
                            [-0.17279032, -0.02163247],
                            [-0.11595435, -0.22871207]])
    np.testing.assert_allclose(a.numpy(), expected_a, atol=1e-5)
    np.testing.assert_allclose(d.numpy(), expected_d, atol=1e-5)
