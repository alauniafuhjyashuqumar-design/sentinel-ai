import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model():
    # 1. تحديد المسارات
    model_path = os.path.join('MODELS', 'fraud_model.pkl')
    data_path = os.path.join('DATA', 'transactions.csv')
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("❌ خطأ: ملف النموذج أو البيانات غير موجود في المسار المحدد.")
        return

    # 2. تحميل الموديل والبيانات
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    # تحديد الميزات (Features) والهدف (Target) بناءً على تدريب المشروع
    # نفترض أن العمود الأخير أو عمود 'is_fraud' هو الهدف
    target_col = 'is_fraud' if 'is_fraud' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 3. التنبؤ وحساب المقاييس
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    print("="*50)
    print("📊 تقرير أداء نموذج SENTINEL_AI الشامل:")
    print("="*50)
    
    print("\n🔹 Confusion Matrix (مصفوفة الارتباك):")
    print(confusion_matrix(y, y_pred))
    
    print("\n🔹 Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(y, y_pred))
    
    try:
        roc_auc = roc_auc_score(y, y_prob)
        print(f"\n🔹 ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print(f"\n⚠️ تعذر حساب ROC-AUC: {e}")
    
    print("="*50)

if __name__ == "__main__":
    evaluate_model()