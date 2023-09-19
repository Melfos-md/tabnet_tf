# Test ghost batchnormalization
import numpy as np
import tensorflow as tf
from models.components.ghost_batchnormalization import GhostBatchNormalization
tf.config.run_functions_eagerly(True)

np.random.seed(42)
tf.random.set_seed(42)

def test_gbn_output_shape():
    gbn = GhostBatchNormalization(virtual_batch_size=32)
    input = tf.random.normal(shape=(128, 64))
    output = gbn(input, training=True)

    assert input.shape == output.shape

def test_ghost_batchnorm_mean_variance():
    """
    Create input data with mean=0 and variance=1.
    Check if batches have the same statistics.
    """
    VIRTUAL_BATCH_SIZE = 32
    EPSILON = 1e-4

    gbn = GhostBatchNormalization(virtual_batch_size=VIRTUAL_BATCH_SIZE)

    input_data = tf.random.normal(shape=(128, 64))

    output = gbn(input_data, training=True)

    splits = tf.split(output, num_or_size_splits=input_data.shape[0] // VIRTUAL_BATCH_SIZE, axis=0)

    for split in splits:
        # Check that mean is close to zero
        assert tf.abs(tf.reduce_mean(split)) < EPSILON, f"Expected mean close to 0, but got {tf.reduce_mean(split)}."

        # Check that variance is close to 1
        variance = tf.reduce_mean(tf.square(split - tf.reduce_mean(split)))
        assert tf.abs(variance - 1) < EPSILON, f"Expected variance close to 1, but got {variance}."



def test_ghost_batchnorm_inference():
    INPUT_SHAPE = 128, 64

    gbn = GhostBatchNormalization(virtual_batch_size=32)

    training_data = tf.random.normal(shape=INPUT_SHAPE)
    inference_data = tf.random.normal(shape=INPUT_SHAPE)

    # Force training by passing `training=True`
    _ = gbn(training_data, training=True)

    # Inference
    inference_output = gbn(inference_data, training=False)

    assert inference_output.shape == INPUT_SHAPE
