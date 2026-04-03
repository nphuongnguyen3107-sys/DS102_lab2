import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess_data.data_loader import load_mnist_data

def main():
    print("="*60)
    print("ASSIGNMENT 3: Using Scikit-Learn for MNIST Classification")
    print("="*60)
    
    # ============================================
    # 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU CHUNG
    # ============================================
    print("\n[INFO] Đang tải dữ liệu...")
    # Lưu ý: Sửa data_path tùy theo thư mục của bạn (ví dụ: data_path='.')
    train_images, train_labels, test_images, test_labels = load_mnist_data(data_path='data')
    
    # Làm phẳng (flatten) và chuẩn hóa (normalize)
    # Không cần thêm bias vì sklearn tự động xử lý (fit_intercept=True)
    X_train_full = train_images.reshape(train_images.shape[0], -1) / 255.0
    X_test_full = test_images.reshape(test_images.shape[0], -1) / 255.0
    y_train_full = train_labels
    y_test_full = test_labels

    # ============================================
    # TASK 1: BINARY LOGISTIC REGRESSION (Digits 0 vs 1)
    # ============================================
    print("\n" + "-"*60)
    print("TASK 1: BINARY LOGISTIC REGRESSION (Digits 0 vs 1)")
    print("-"*60)
    
    # Lọc dữ liệu chỉ lấy số 0 và 1
    train_mask = np.isin(y_train_full, [0, 1])
    test_mask = np.isin(y_test_full, [0, 1])
    
    X_train_bin = X_train_full[train_mask]
    y_train_bin = y_train_full[train_mask]
    X_test_bin = X_test_full[test_mask]
    y_test_bin = y_test_full[test_mask]
    
    print(f"Dữ liệu huấn luyện (Binary): {X_train_bin.shape}")
    print("Đang huấn luyện mô hình Logistic Regression...")
    
    # Khởi tạo và huấn luyện mô hình
    # max_iter=1000 để đảm bảo thuật toán hội tụ
    binary_model = LogisticRegression(max_iter=1000, random_state=42)
    binary_model.fit(X_train_bin, y_train_bin)
    
    # Dự đoán và đánh giá
    y_pred_bin = binary_model.predict(X_test_bin)
    acc_bin = accuracy_score(y_test_bin, y_pred_bin)
    
    print(f"\nKết quả Test (Binary):")
    print(f"Accuracy: {acc_bin:.4f} ({acc_bin*100:.2f}%)")
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test_bin, y_pred_bin))

    # ============================================
    # TASK 2: SOFTMAX REGRESSION (Digits 0-9)
    # ============================================
    print("\n" + "-"*60)
    print("TASK 2: SOFTMAX REGRESSION (Digits 0-9)")
    print("-"*60)
    
    print(f"Dữ liệu huấn luyện (Multiclass): {X_train_full.shape}")
    print("Đang huấn luyện mô hình Softmax Regression...")
    
    # Khởi tạo và huấn luyện mô hình
    # multi_class='multinomial' chính là cấu hình để chạy Softmax Regression
    softmax_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    softmax_model.fit(X_train_full, y_train_full)
    
    # Dự đoán và đánh giá
    y_pred_multi = softmax_model.predict(X_test_full)
    acc_multi = accuracy_score(y_test_full, y_pred_multi)
    
    print(f"\nKết quả Test (Softmax):")
    print(f"Accuracy: {acc_multi:.4f} ({acc_multi*100:.2f}%)")
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test_full, y_pred_multi))

    print("\n" + "="*60)
    print("✅ ASSIGNMENT 3 HOÀN THÀNH!")
    print("="*60)

if __name__ == "__main__":
    main()