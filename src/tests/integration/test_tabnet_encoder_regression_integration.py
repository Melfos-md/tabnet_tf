import tensorflow as tf
import numpy as np
from models.tabnet_encoder import TabNetEncoder

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def generate_synthetic_data(num_samples=1000, num_features=10):
    """Generate synthetic data for regression."""
    X = np.random.rand(num_samples, num_features)
    # Simple linear relation with some noise
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, size=num_samples)
    return X, y

def test_tabnet_encoder_integration():
    # Generate synthetic data
    X, y = generate_synthetic_data()
    
    # Split data into training and validation sets
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Hyperparameters
    N_step = 4
    num_features = X_train.shape[1]
    N_a = N_d = 2
    virtual_batch_size = 32
    
    # Instantiate the TabNetEncoder
    model = TabNetEncoder(target_is_discrete=False, N_step=N_step, num_features=num_features, 
                          N_a=N_a, N_d=N_d, virtual_batch_size=virtual_batch_size, seed=SEED)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    
    # Evaluate the model
    mse = model.evaluate(X_val, y_val)
    
    # Assert that the MSE is below a certain threshold
    assert mse < 0.1, f"Expected MSE < 0.1 but got {mse}"