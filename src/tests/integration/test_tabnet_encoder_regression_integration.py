import tensorflow as tf
import numpy as np
from models.tabnet_encoder import TabNetEncoder
from tests.utils import generate_synthetic_data
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def test_tabnet_encoder_regression_integration_1():

    # Hyperparameters
    batch_size = 512
    N_a = N_d = 8
    num_features = 64
    data_size = batch_size * 3
    output_units = 1

    # Generate random input features and targets
    features = tf.constant(np.random.rand(data_size, num_features), dtype=tf.float32)
    targets = tf.constant(np.random.rand(data_size, output_units), dtype=tf.float32)

    # Initialize the TabNetEncoder model
    model = TabNetEncoder(target_is_discrete=False, output_units=output_units, N_step=4, batch_size=batch_size, num_features=num_features, N_a=N_a, N_d=N_d, seed=SEED)

    # Compile the model with a regression loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model for a few epochs
    history = model.fit(features, targets, epochs=5, batch_size=batch_size)

    # Check if the loss decreased
    assert history.history['loss'][0] > history.history['loss'][-1], "Model did not train properly."

    # Predict with the model using the appropriate batch size
    predictions = model.predict(features, batch_size=batch_size)

    # Check the shape of the predictions
    assert predictions.shape == (data_size, output_units), f"Expected predictions shape {(batch_size, output_units)}, but got {predictions.shape}"



def test_tabnet_encoder_regression_integration_2():
    batch_size = 256
    num_features = 10
    num_samples = batch_size * 10
    N_step = 4
    N_a = N_d = 2
    virtual_batch_size = 32
    
    # Generate synthetic data
    X, y = generate_synthetic_data(num_samples=num_samples, num_features=num_features, seed=SEED) # 1280 = 1024 + 256 (train + val)
    
    # Split data into training and validation sets
    split = 8 * batch_size
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Instantiate the TabNetEncoder
    model = TabNetEncoder(target_is_discrete=False, N_step=N_step, batch_size=batch_size, num_features=num_features, 
                          N_a=N_a, N_d=N_d, virtual_batch_size=virtual_batch_size, seed=SEED)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_val, y_val))
    
    # Evaluate the model
    mse = model.evaluate(X_val, y_val, batch_size=batch_size)
    
    # Assert that the MSE is below a certain threshold
    assert mse < 0.1, f"Expected MSE < 0.1 but got {mse}"

