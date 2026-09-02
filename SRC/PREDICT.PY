from click import File
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def predict_transaction(user_id, device_id, amount, failed_logins, location_change):
    # بيانات تجريبية سابقة لغرض التدريب السريع داخل الكود أو نقدر نحفظ الموديل بملف ونستدعيه
    # لسهولة التنفيذ، خلينا ندرب نموذج مصغر ونختبره مباشرة:
    file_path = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'raw', 'transactions.csv')
    df = pd.read_csv(file_path)
    
    le_user = LabelEncoder()
    le_device = LabelEncoder()
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
    df['device_id_encoded'] = le_device.fit_transform(df['device_id'])
    
    X = df[['user_id_encoded', 'device_id_encoded', 'amount', 'failed_logins', 'location_change']]
    y = df['is_fraud']
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    # معالجة المدخلات الجديدة (بافتراض أن المستخدم أو الجهاز قد يكون جديداً، نستخدم transform بحذر أو نتأكد)
    try:
        u_encoded = le_user.transform([user_id])[0]
    except ValueError:
        u_encoded = 0 # قيمة افتراضية إذا كان المستخدم جديد كلياً
        
    try:
        d_encoded = le_device.transform([device_id])[0]
    except ValueError:
        d_encoded = 0

    sample = [[u_encoded, d_encoded, amount, failed_logins, location_change]]
    prediction = model.predict(sample)
    
    if prediction[0] == 1:
        print("\n🚨 [WARNING] تحذير خطير: هذه المعاملة [احتيالية] وتم حجزها بنجاح!")
    else:
        print("\n✅ [SAFE] المعاملة سليمة وطبيعية، مبروك المرور!")
predict_transaction("user_50", "device_10", 3500.0, 4, 1)
