import os
import joblib
import pandas as pd

def predict_transaction(amount, old_balance, new_balance):
    # تحديد مسار النموذج المحفوظ بدقة
    model_path = os.path.join(os.path.dirname(__file__), '..', 'MODELS', 'fraud_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found! Please train the model first.")
        
    # تحميل النموذج
    model = joblib.load(model_path)
    
    # تجهيز البيانات كـ DataFrame بنفس أسماء الأعمدة المستخدمة في التدريب
    input_data = pd.DataFrame([[amount, old_balance, new_balance]], 
                              columns=['amount', 'oldbalanceOrg', 'newbalanceDest'])
    
    # التنبؤ بالنتيجة والاحتمالية (Risk Score)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # احتمالية الاحتيال (0 إلى 1)
    
    risk_score = round(probability * 100, 2) # تحويلها إلى نسبة مئوية
    
    # تحديد مستويات الخطورة بشكل علمي ومنطقي
    if risk_score < 30:
        risk_level = "Low"
    elif risk_score < 60:
        risk_level = "Medium"
    elif risk_score < 85:
        risk_level = "High"
    else:
        risk_level = "Critical"
        
    # إضافة تفسير منطقي (Explainability) لعرضه للمستخدم وللدكتور
    reasons = []
    if amount > 5000:
        reasons.append("مبلغ المعاملة مرتفع جداً مقارنة بالمعايير المعتادة.")
    if old_balance < 500:
        reasons.append("الرصيد المتاح في الحساب المرسل قليل جداً ولا يغطي قيمة المعاملة بشكل آمن.")
    if prediction == 1 and not reasons:
        reasons.append("سلوك المعاملة يتطابق مع الأنماط الاحتيالية المكتشفة في بيانات التدريب.")
    elif prediction == 0:
        reasons.append("المعاملة تقع ضمن النطاق الطبيعي للعمليات المالية الآمنة.")
        
    return {
        "is_fraud": int(prediction),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reasons": reasons
    }

if __name__ == "__main__":
    # اختبار تجريبي سريع للتنبؤ بمعاملة مشبوهة
    print("[*] Testing prediction with a sample suspicious transaction...")
    sample_result = predict_transaction(15000, 50, -14950)
    print("Test Result:", sample_result)