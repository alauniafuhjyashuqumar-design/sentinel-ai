import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def train_fraud_model():
    print("[*] Loading dataset for training...")
    
    # تحديد مسار البيانات بدقة
    file_path = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'transactions.csv')
    df = pd.read_csv(file_path)
    
    # تحديد الـ Features (المتغيرات المستقلة) والـ Target (الهدف)
    # الاعتماد على الأعمدة الموجودة في الجدول
    X = df[['amount', 'oldbalanceOrg', 'newbalanceDest']]
    y = df['is_fraud']
    
    # تقسيم البيانات إلى تدريب (80%) واختبار (20%) مع الحفاظ على نسبة التوزيع (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("[*] Training Random Forest Classifier with class balancing...")
    # إضافة class_weight='balanced' لحل مشكلة عدم توازن البيانات
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("[*] Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # طباعة نتائج التقييم التفصيلية لعرضها أمام الدكتور
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print("Could not calculate ROC-AUC:", e)
        
    # حفظ النموذج المدرب في مجلد MODELS
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'MODELS'), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'MODELS', 'fraud_model.pkl')
    joblib.dump(model, model_path)
    print(f"[*] Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_fraud_model()