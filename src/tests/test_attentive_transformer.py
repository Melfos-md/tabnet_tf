import numpy as np
import pytest
import tensorflow as tf
from models.attentive_transformer import AttentiveTransformer

def test_output_shape():
    input_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
    model = AttentiveTransformer()
    output = model(input_tensor)

    assert output.shape == input_tensor.shape

def test_mask_creation():
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = AttentiveTransformer(seed=seed)
    input_tensor = tf.constant(np.random.rand(2,3))
    output = model(input_tensor).numpy()

    expected_output = np.array([
        [0.53271145, 0.439279, 0.02800953],
        [0.44851196, 0.3642257, 0.18726212]
    ])

    assert np.all(np.allclose(output, expected_output, atol=1e-5))

def test_incorrect_shape():
    input = tf.constant(np.random.rand(4,2,30))
    model = AttentiveTransformer()

    with pytest.raises(ValueError):
        output = model(input)

