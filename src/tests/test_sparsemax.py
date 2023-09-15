import tensorflow as tf
import numpy as np
from models.components.sparsemax import Sparsemax

# Check if output sum is 1
def test_sparsemax_sum():
    layer = Sparsemax()
    logits = tf.constant([[1.0, 2.0, 3.0]])
    output = layer(logits).numpy()

    assert np.all(np.isclose(np.sum(output), 1.0, atol=1e-5))
    print("Sparsemax sum test passed!")

# 0 vector should give 0s
def test_sparsemax_zeros():
    layer = Sparsemax()
    logits = tf.constant([[0.0, 0.0, 0.0]])
    output = layer(logits).numpy()

    assert np.all(output >= 0.0)
    assert np.isclose(np.sum(output), 1.0)
    print("Sparsemax zeros test passed")

def test_sparsemax_regression():
    np.random.seed(42)
    layer = Sparsemax()

    logits = tf.constant(np.random.rand(1, 5))
    output = layer(logits).numpy()

    expected_output = np.array([[0.0 , 0.523592, 0.30487165, 0.17153624, 0.0]])

    assert np.all(np.allclose(output, expected_output, atol=1e-5))
    print("Sparsemax regression test passed!")