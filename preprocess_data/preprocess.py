import numpy as np

#Lọc dữ liệu theo nhãn
def filter_by_labels(images, labels, target_labels):
    #Tạo mask để lọc các mẫu có nhãn thuộc target_labels
    mask = np.isin(labels, target_labels)
    #Trả về các mẫu và nhãn đã lọc
    return images[mask], labels[mask]

#Tiền xử lý cho bài toán phân loại nhị phân (chỉ giữ lại các mẫu có nhãn 0 và 1)
def preprocess_binary(images, labels, normalize=True, add_bias=True):
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

#Hàm one-hot encoding cho nhãn đa lớp
def one_hot_encode(labels, num_classes=10):
    N = len(labels)
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), labels.flatten()] = 1
    return one_hot

#Tiền xử lý cho bài toán phân loại đa lớp (giữ lại tất cả các mẫu, chỉ one-hot encode nhãn)
def preprocess_multiclass(images, labels, num_classes=10, normalize=True, add_bias=True):
    X = images.reshape(images.shape[0], -1)
    if normalize:
        X = X / 255.0
    if add_bias:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    y_onehot = one_hot_encode(labels, num_classes)
    return X, y_onehot