import numpy as np
import idx2numpy

#Hàm để tải dữ liệu MNIST từ các file .idx
def load_mnist_data(data_path='data'):
    train_images = idx2numpy.convert_from_file(f"{data_path}/train-images.idx3-ubyte")
    train_labels = idx2numpy.convert_from_file(f"{data_path}/train-labels.idx1-ubyte")
    test_images = idx2numpy.convert_from_file(f"{data_path}/t10k-images.idx3-ubyte")
    test_labels = idx2numpy.convert_from_file(f"{data_path}/t10k-labels.idx1-ubyte")
    
    print(f"Loaded MNIST data:")
    print(f"  Train: {train_images.shape[0]} images, shape {train_images.shape[1:]}")
    print(f"  Test: {test_images.shape[0]} images, shape {test_images.shape[1:]}")
    
    return train_images, train_labels, test_images, test_labels