import numpy as np

def calculate_binary_metrics(y_true, y_pred):
    """
    Calculate precision, recall, f1 for binary classification
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }

def calculate_multiclass_metrics(y_true, y_pred, num_classes=10):
    """
    Calculate precision, recall, f1 for multiclass classification
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    return {
        'macro': {
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro')
        },
        'micro': {
            'precision': precision_score(y_true, y_pred, average='micro'),
            'recall': recall_score(y_true, y_pred, average='micro'),
            'f1': f1_score(y_true, y_pred, average='micro')
        }
    }