import numpy as np

def filter_by_labels(images, labels, target_labels):
    """
    Filter data by target labels
    """
    mask = np.isin(labels, target_labels)
    return images[mask], labels[mask]

def preprocess_binary(images, labels, normalize=True, add_bias=True):
    """
    Preprocess data for binary classification (0 vs 1)
    """
    # Filter only 0 and 1
    X, y = filter_by_labels(images, labels, [0, 1])
    
    # Flatten images
    N = X.shape[0]
    X = X.reshape(N, -1)
    
    # Normalize
    if normalize:
        X = X / 255.0
    
    # Add bias
    if add_bias:
        X = np.hstack([np.ones((N, 1)), X])
    
    # Reshape labels
    y = y.reshape(-1, 1)
    
    return X, y

def one_hot_encode(labels, num_classes=10):
    """
    Convert labels to one-hot encoding
    """
    N = len(labels)
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), labels.flatten()] = 1
    return one_hot

def preprocess_multiclass(images, labels, num_classes=10, normalize=True, add_bias=True):
    """
    Preprocess data for multiclass classification (0-9)
    """
    # Flatten images
    X = images.reshape(images.shape[0], -1)
    
    # Normalize
    if normalize:
        X = X / 255.0
    
    # Add bias
    if add_bias:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
    # One-hot encode labels
    y_onehot = one_hot_encode(labels, num_classes)
    
    return X, y_onehot