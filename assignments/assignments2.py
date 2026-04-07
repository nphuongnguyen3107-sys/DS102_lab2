import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from preprocess_data import load_mnist_data, preprocess_multiclass, calculate_multiclass_metrics

# Mô hình Softmax Regression cho bài toán phân loại đa lớp (10 lớp số 0-9)
class SoftmaxRegression:
    def __init__(self, epoch=500, lr=0.01, num_classes=10):
        self.epoch = epoch              
        self.lr = lr                     
        self.num_classes = num_classes  
        self.weights = None              
        self.losses = []    

    # Hàm kích hoạt Softmax: Biến đổi mảng giá trị đầu ra (Z) thành phân bố xác suất.    
    def softmax(self, z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # Hàm huấn luyện mô hình (Tìm ma trận trọng số W tối ưu)
    def fit(self, X, y_onehot):
        num_samples, num_features = X.shape
        self.weights = np.zeros((num_features, self.num_classes))
        for _ in range(self.epoch):
            linear_model = np.dot(X, self.weights)
            y_predicted = self.softmax(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y_onehot))
            self.weights -= self.lr * dw
            epsilon = 1e-9 # Thêm epsilon nhỏ vào log để tránh lỗi log(0)
            loss = -np.mean(np.sum(y_onehot * np.log(y_predicted + epsilon), axis=1))
            self.losses.append(loss)

    # Hàm dự đoán nhãn cho dữ liệu mới
    def predict(self, X):
        linear_model = np.dot(X, self.weights)
        y_predicted_prob = self.softmax(linear_model)
        y_predicted_cls = np.argmax(y_predicted_prob, axis=1)
        return y_predicted_cls

# Hàm chính để chạy toàn bộ quy trình
def main():
    print("="*60)
    print("ASSIGNMENT 2: Softmax Regression (Digits 0-9)")
    print("="*60)
    print("\n Đang tải dữ liệu MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist_data()

    print("\n Đang tiền xử lý dữ liệu (Multiclass & One-Hot Encoding)...")
    X_train, y_train_onehot = preprocess_multiclass(train_images, train_labels, num_classes=10)
    X_test, y_test_onehot = preprocess_multiclass(test_images, test_labels, num_classes=10)
    
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Kích thước tập kiểm tra: {X_test.shape}")
    print("\n[INFO] Bắt đầu huấn luyện Softmax Regression...")
    model = SoftmaxRegression(epoch=500, lr=0.01, num_classes=10)
    model.fit(X_train, y_train_onehot)
    print("-> Huấn luyện xong!")
    plt.figure(figsize=(10, 6))
    plt.plot(model.losses, color='red', linewidth=2)
    plt.xlabel('Vòng lặp (Epoch)', fontsize=12)
    plt.ylabel('Hàm suy hao (Cross-Entropy Loss)', fontsize=12)
    plt.title('Biểu đồ huấn luyện - Softmax Regression (0-9)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    print("\n Đang đánh giá trên tập Test...")
    y_pred = model.predict(X_test) # Lúc này y_pred đã được argmax chuyển về dạng số nguyên (0->9)
    metrics = calculate_multiclass_metrics(test_labels, y_pred)
    
    print(f"\nKẾT QUẢ TEST:")
    print("-" * 40)
    print(f"  Độ chính xác (Accuracy):   {np.mean(y_pred == test_labels):.4f}")
    
    if 'macro' in metrics:
        print(f"  Macro Avg - Precision:     {metrics['macro']['precision']:.4f}")
        print(f"  Macro Avg - Recall:        {metrics['macro']['recall']:.4f}")
        print(f"  Macro Avg - F1-Score:      {metrics['macro']['f1']:.4f}")

    # 6. In thử một vài kết quả dự đoán (10 mẫu đầu tiên)
    print("\n--- Dự đoán trực quan (10 ảnh đầu tiên) ---")
    for i in range(min(10, len(X_test))):
        true_label = test_labels[i]
        pred_label = y_pred[i]
        status = "Đúng" if true_label == pred_label else "Sai"
        print(f"  Ảnh số {i+1:2d}: Thực tế = {true_label} | Dự đoán = {pred_label} {status}")

    print("\n" + "="*60)
    print("ASSIGNMENT 2 HOÀN THÀNH!")
    print("="*60)

if __name__ == "__main__":
    main()