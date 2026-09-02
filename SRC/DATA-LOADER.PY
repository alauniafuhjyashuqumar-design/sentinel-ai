import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    print("[*] Loading raw dataset for preprocessing...")
    
    # تحديد مسار البيانات الخام
    file_path = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'raw', 'transactions.csv')
    df = pd.read_csv(file_path)
    
    # معالجة الأعمدة النصية وتحويلها إلى قيم رقمية (Encoding)
    print("[*] Encoding categorical features...")
    le_user = LabelEncoder()
    le_device = LabelEncoder()
    
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
    df['device_id_encoded'] = le_device.fit_transform(df['device_id'])
    
    # تحديد الميزات (Features - X) والهدف (Target - y)
    feature_columns = ['user_id_encoded', 'device_id_encoded', 'amount', 'failed_logins', 'location_change']
    X = df[feature_columns]
    y = df['is_fraud']
    
    # تقسيم البيانات إلى مجموعة تدريب (Train) ومجموعة اختبار (Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[+] Data successfully preprocessed and split!")
    print(f"    - Training set shape: {X_train.shape}")
    print(f"    - Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data()
    