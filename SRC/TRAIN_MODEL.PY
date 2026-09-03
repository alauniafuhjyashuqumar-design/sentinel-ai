import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_fraud_model():
    print("🔄 جاري فحص البيانات والملفات...")
    
    # إنشاء المجلدات تلقائياً إذا مو موجودة
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    data_path = "data/transactions.csv"
    
    # إذا ملف البيانات مو موجود، ننشئ بيانات تجريبية ذكية وصحيحة تلقائياً!
    if not os.path.exists(data_path):
        print("⚙️ جاري توليد بيانات المعاملات التجريبية تلقائياً...")
        np.random.seed(42)
        n_samples = 1000
        synthetic_data = {
            'amount': np.random.exponential(scale=150, size=n_samples),
            'oldbalanceOrg': np.random.uniform(1000, 60000, size=n_samples),
            'newbalanceDest': np.random.uniform(0, 60000, size=n_samples),
            'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.94, 0.06])
        }
        df = pd.DataFrame(synthetic_data)
        df.to_csv(data_path, index=False)
        print("✅ تم إنشاء ملف البيانات بنجاح في مجلد data.")
    else:
        df = pd.read_csv(data_path)

    # فصل الميزات (Features) عن عمود الاحتيال (Target)
    if 'is_fraud' in df.columns:
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # تقسيم البيانات للتدريب والاختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # تدريب النموذج
    print("🚀 جاري تدريب نموذج كشف الاحتيال (SENTINEL AI)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # تقييم النموذج
    predictions = model.predict(X_test)
    print("📊 تقرير أداء النموذج:")
    print(classification_report(y_test, predictions))

    # حفظ النموذج المدرب
    model_path = "models/fraud_model.pkl"
    joblib.dump(model, model_path)
    print(f"🎉 تم حفظ النموذج المدرب بنجاح في المسار: {model_path}")

if __name__ == "__main__":
    train_fraud_model()
    