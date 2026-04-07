import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import load_mnist_data, preprocess_binary, calculate_binary_metrics

class LogisticRegression:
    def __init__(self, epoch=500, lr=0.01):
        self.epoch = epoch        
        self.lr = lr               
        self.weights = None       
        self.losses = []          

    # Hàm kích hoạt Sigmoid: Biến đổi giá trị đầu ra (z) thành xác suất trong khoảng (0, 1)    
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))
    
    # Hàm huấn luyện mô hình (Tìm trọng số w tối ưu)
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros((num_features, 1))
        for _ in range(self.epoch):
            linear_model = np.dot(X, self.weights)
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            self.weights -= self.lr * dw
            epsilon = 1e-9
            loss = -np.mean(y * np.log(y_predicted + epsilon) + (1 - y) * np.log(1 - y_predicted + epsilon))
            self.losses.append(loss)

    # Hàm dự đoán nhãn (0 hoặc 1) dựa trên ngưỡng 0.5
    def predict(self, X):
        linear_model = np.dot(X, self.weights)
        y_predicted = self.sigmoid(linear_model)
        
        y_predicted_cls = np.where(y_predicted > 0.5, 1, 0)
        return y_predicted_cls

# Hàm chính để chạy toàn bộ quy trình
def main():
    print("="*60)
    print("ASSIGNMENT 1: Binary Logistic Regression (Digits 0 vs 1)")
    print("="*60)
    
    print("\n Đang tải dữ liệu MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    print("\nĐang tiền xử lý dữ liệu (Binary)...")
    X_train, y_train = preprocess_binary(train_images, train_labels)
    X_test, y_test = preprocess_binary(test_images, test_labels)
    
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Kích thước tập kiểm tra: {X_test.shape}")
    
    print("\n[INFO] Bắt đầu huấn luyện Logistic Regression...")
    model = LogisticRegression(epoch=500, lr=0.01)
    model.fit(X_train, y_train)
    print("-> Huấn luyện xong!")
  
    plt.figure(figsize=(10, 6))
    plt.plot(model.losses, color='blue', linewidth=2)
    plt.xlabel('Vòng lặp (Epoch)', fontsize=12)
    plt.ylabel('Hàm suy hao (Loss)', fontsize=12)
    plt.title('Biểu đồ huấn luyện - Hồi quy Logistic Nhị phân', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    print("\n[INFO] Đang đánh giá trên tập Test...")
    y_pred = model.predict(X_test)
    metrics = calculate_binary_metrics(y_test, y_pred)
    
    print(f"\nKẾT QUẢ TEST:")
    print("-" * 30)
    print(f"  Accuracy (Độ chính xác):     {metrics['accuracy']:.4f}")  
    print(f"  Precision (Độ xác thực):     {metrics['precision']:.4f}") 
    print(f"  Recall (Độ bao phủ):         {metrics['recall']:.4f}")    
    print(f"  F1-Score:                    {metrics['f1']:.4f}")        
    print("-" * 30)

if __name__ == "__main__":
    main()