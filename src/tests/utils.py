import numpy as np

def generate_synthetic_data(num_samples=1024, num_features=10, seed=None):
    """Generate synthetic data for regression."""
    np.random.seed(seed)
    X = np.random.rand(num_samples, num_features)
    # Simple linear relation with some noise
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, size=num_samples)
    return X, y
