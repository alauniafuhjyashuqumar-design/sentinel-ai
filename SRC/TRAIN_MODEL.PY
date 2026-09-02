import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_fraud_model():
    print("[*] Loading and preparing data for training...")
    
    # تحميل البيانات المعالجة مباشرة
    file_path = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'raw', 'transactions.csv')
    df = pd.read_csv(file_path)
    
    # تشفير البيانات النصية
    le_user = LabelEncoder()
    le_device = LabelEncoder()
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
    df['device_id_encoded'] = le_device.fit_transform(df['device_id'])
    
    # تحديد الميزات والهدف
    feature_columns = ['user_id_encoded', 'device_id_encoded', 'amount', 'failed_logins', 'location_change']
    X = df[feature_columns]
    y = df['is_fraud']
    
    # التقسيم (Train/Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[*] Training Random Forest Classifier with balanced weights...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # تقييم النموذج على بيانات الاختبار
    print("[*] Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    print(f"\n[+] Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n[📊] Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    train_fraud_model()
    