import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
# Import các module sức mạnh từ thư viện Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Import hàm tải dữ liệu do bạn tự viết
from preprocess_data import load_mnist_data

def main():
    print("="*60)
    print("ASSIGNMENT 3: Using Scikit-Learn for MNIST Classification")
    print("="*60)
    
    print("\n[INFO] Đang tải dữ liệu MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    print("[INFO] Đang làm phẳng (Flatten) và chuẩn hóa (Normalize) dữ liệu...")
    X_train_full = train_images.reshape(train_images.shape[0], -1) / 255.0
    X_test_full = test_images.reshape(test_images.shape[0], -1) / 255.0
    y_train_full = train_labels
    y_test_full = test_labels

    # =====================================================================
    # TASK 1: BINARY LOGISTIC REGRESSION (Phân loại số 0 và 1)
    # =====================================================================
    print("\n" + "-"*60)
    print("TASK 1: BINARY LOGISTIC REGRESSION (Digits 0 vs 1)")
    print("-"*60)
    
    # Tạo mặt nạ boolean để lọc riêng các ảnh có nhãn là 0 và 1
    train_mask = np.isin(y_train_full, [0, 1])
    test_mask = np.isin(y_test_full, [0, 1])
    
    # Áp mặt nạ vào tập dữ liệu lớn để trích xuất ra tập dữ liệu nhị phân
    X_train_bin = X_train_full[train_mask]
    y_train_bin = y_train_full[train_mask]
    X_test_bin = X_test_full[test_mask]
    y_test_bin = y_test_full[test_mask]
    
    print(f"Kích thước tập huấn luyện (Binary): {X_train_bin.shape}")
    print("[INFO] Đang huấn luyện mô hình Logistic Regression (Nhị phân)...")
    
    binary_model = LogisticRegression(max_iter=1000, random_state=42)

    binary_model.fit(X_train_bin, y_train_bin)
    
    # Dự đoán trên tập Test
    y_pred_bin = binary_model.predict(X_test_bin)
    acc_bin = accuracy_score(y_test_bin, y_pred_bin)
    
    # In kết quả chi tiết
    print(f"\nKẾT QUẢ TEST (BINARY):")
    print(f"  -> Accuracy: {acc_bin:.4f} ({acc_bin*100:.2f}%)")
    print("\nBáo cáo chi tiết (Classification Report):")
    print(classification_report(y_test_bin, y_pred_bin))


    # =====================================================================
    # TASK 2: SOFTMAX REGRESSION (Phân loại đa lớp từ 0 đến 9)
    # =====================================================================
    print("\n" + "-"*60)
    print("TASK 2: SOFTMAX REGRESSION / MULTINOMIAL (Digits 0-9)")
    print("-"*60)
    
    print(f"Kích thước tập huấn luyện (Multiclass): {X_train_full.shape}")
    print("[INFO] Đang huấn luyện mô hình Softmax Regression (Đa lớp)...")
    
    softmax_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    
    softmax_model.fit(X_train_full, y_train_full)
    
    # Dự đoán
    y_pred_multi = softmax_model.predict(X_test_full)
    acc_multi = accuracy_score(y_test_full, y_pred_multi)
    
    # In kết quả
    print(f"\nKẾT QUẢ TEST (SOFTMAX):")
    print(f"  -> Accuracy: {acc_multi:.4f} ({acc_multi*100:.2f}%)")
    print("\nBáo cáo chi tiết (Classification Report):")
    print(classification_report(y_test_full, y_pred_multi))

    print("\n" + "="*60)
    print(" ASSIGNMENT 3 HOÀN THÀNH!")
    print("="*60)

if __name__ == "__main__":
    main()