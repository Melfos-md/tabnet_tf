import tensorflow as tf
import numpy as np
from models.gated_linear_unit import GLU

np.random.seed(42)
tf.random.set_seed(42)

def test_glu_output_shape():
    batch_size, num_features = 5, 10
    x = tf.constant(np.random.rand(batch_size, num_features))
    glu = GLU()
    output = glu(x)

    assert output.shape == x.shape

def test_value():
    input_shape = 5, 10
    x = tf.ones(input_shape)
    output = GLU()(x).numpy()
    expected_output = np.full(input_shape, 0.7310586)

    assert np.allclose(output, expected_output, atol=1e-5)

def test_glu_asymmetry():
    negative_tensor = tf.constant(np.full((3,4), -1.0), dtype=tf.float32)
    positive_tensor = tf.constant(np.full((3,4), 1.0), dtype=tf.float32)

    glu = GLU()
    negative_output = glu(negative_tensor).numpy()
    positive_output = glu(positive_tensor).numpy()

    assert not np.array_equal(negative_output, positive_output)

def test_glu_gradient():
    input_tensor = tf.keras.Input(shape=(4,))
    output_tensor = GLU()(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam', loss='mse')

    random_tensor = tf.random.normal((3,4))

    with tf.GradientTape() as tape:
        tape.watch(random_tensor)
        output = model(random_tensor)
        loss = tf.reduce_mean(output) # simple operation to get a scalar value
    
    grads = tape.gradient(loss, random_tensor)

    assert grads is not None, "Gradient is None"
    assert not tf.math.reduce_all(grads == 0), "All gradient values are zero!"