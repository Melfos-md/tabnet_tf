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
    output_units = 1

    # Generate random input features and targets
    features = tf.constant(np.random.rand(batch_size, num_features), dtype=tf.float32)
    targets = tf.constant(np.random.rand(batch_size, output_units), dtype=tf.float32)

    # Initialize the TabNetEncoder model
    model = TabNetEncoder(target_is_discrete=False, output_units=output_units, N_step=4, num_features=num_features, N_a=N_a, N_d=N_d, seed=SEED)

    # Compile the model with a regression loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model for a few epochs
    history = model.fit(features, targets, epochs=5, batch_size=batch_size)

    # Check if the loss decreased
    assert history.history['loss'][0] > history.history['loss'][-1], "Model did not train properly."

    # Predict with the model using the appropriate batch size
    predictions = model.predict(features, batch_size=batch_size)

    # Check the shape of the predictions
    assert predictions.shape == (batch_size, output_units), f"Expected predictions shape {(batch_size, output_units)}, but got {predictions.shape}"



def test_tabnet_encoder_regression_integration_2():

    
    # Generate synthetic data
    X, y = generate_synthetic_data(num_samples=1280, num_features=10, seed=SEED) # 1280 = 1024 + 256 (train + val)
    
    # Split data into training and validation sets
    X_train, X_val = X[:1024], X[1024:]
    y_train, y_val = y[:1024], y[1024:]

    # Hyperparameters
    N_step = 4
    num_features = X_train.shape[1]
    N_a = N_d = 2
    virtual_batch_size = 32
    batch_size = 32

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    

    # Instantiate the TabNetEncoder
    model = TabNetEncoder(target_is_discrete=False, N_step=N_step, num_features=num_features, 
                          N_a=N_a, N_d=N_d, virtual_batch_size=virtual_batch_size, seed=SEED)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(dataset, epochs=10, batch_size=batch_size) #, validation_data=(X_val, y_val)
    
    # Evaluate the model
    #mse = model.evaluate(val_dataset)
    
    # Assert that the MSE is below a certain threshold
    #assert mse < 0.1, f"Expected MSE < 0.1 but got {mse}"

