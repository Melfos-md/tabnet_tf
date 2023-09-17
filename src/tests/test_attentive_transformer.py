import numpy as np
import pytest
import tensorflow as tf
from models.attentive_transformer import AttentiveTransformer

def test_output_shape():
    batch_size = 2
    num_features = 3
    input_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32) # (2,3) = (batch_size, num_features)
    output = AttentiveTransformer(num_features=num_features)(input_tensor)

    assert output.shape.as_list() == [batch_size, num_features]

def test_mask_creation():
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = AttentiveTransformer(num_features=3, seed=seed)
    input_tensor = tf.constant(np.random.rand(2,3))
    output = model(input_tensor).numpy()

    expected_output = np.array([
        [0.53271145, 0.439279, 0.02800953],
        [0.44851196, 0.3642257, 0.18726212]
    ])

    assert np.all(np.allclose(output, expected_output, atol=1e-5))

def test_incorrect_shape():
    input = tf.constant(np.random.rand(4,2,30))
    model = AttentiveTransformer(num_features=2)

    with pytest.raises(ValueError):
        _ = model(input)


def test_prior_scales_update():
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = AttentiveTransformer(num_features=3, relaxation_factor=0.9, seed=seed)
    input_tensor = tf.constant(np.random.rand(2, 3), dtype=tf.float32)
    initial_prior_scales = np.ones((2,3))

    initial_mask = model(input_tensor).numpy()

    updated_mask = model(input_tensor).numpy()

    expected_updated_prior_scales = initial_prior_scales * (0.9 - initial_mask)
    expected_updated_prior_scales *= (0.9 - updated_mask)
    np.testing.assert_allclose(model.prior_scales.numpy(), expected_updated_prior_scales, atol=1e-5)
