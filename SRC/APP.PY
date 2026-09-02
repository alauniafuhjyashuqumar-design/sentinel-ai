import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="SENTINEL_AI Dashboard", page_icon="🛡️", layout="centered")

st.title("🛡️ SENTINEL_AI: نظام كشف الاحتيال الذكي")
st.write("أهلاً بكِ يا نوراية! أدخلي تفاصيل المعاملة أدناه لفحصها فوراً عبر نموذج الذكاء الاصطناعي:")

# تحميل البيانات وتدريب الموديل تلقائياً
@st.cache_resource
def load_model():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'raw', 'transactions.csv')
    if not os.path.exists(file_path):
        return None, None, None
    df = pd.read_csv(file_path)
    
    le_user = LabelEncoder()
    le_device = LabelEncoder()
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
    df['device_id_encoded'] = le_device.fit_transform(df['device_id'])
    
    X = df[['user_id_encoded', 'device_id_encoded', 'amount', 'failed_logins', 'location_change']]
    y = df['is_fraud']
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model, le_user, le_device

model, le_user, le_device = load_model()

if model is None:
    st.error("⚠️ ملف البيانات غير موجود، يرجى التأكد من توليد البيانات أولاً!")
else:
    # تصميم حقول الإدخال في الواجهة
    st.subheader("📝 إدخال بيانات المعاملة الجديدة")
    user_id = st.text_input("معرف المستخدم (User ID)", "user_50")
    device_id = st.text_input("معرف الجهاز (Device ID)", "device_10")
    amount = st.number_input("مبلغ المعاملة ($)", min_value=0.0, value=3500.0, step=50.0)
    failed_logins = st.slider("عدد محاولات الدخول الفاشلة", 0, 10, 4)
    location_change = st.selectbox("هل حدث تغيير في الموقع الجغرافي؟", [0, 1], format_func=lambda x: "نعم (1)" if x == 1 else "لا (0)")

    # زر الفحص والتنبؤ
    if st.button("🔍 فحص المعاملة الآن"):
        try:
            u_encoded = le_user.transform([user_id])[0]
        except ValueError:
            u_encoded = 0 # في حال كان مستخدم جديد كلياً
            
        try:
            d_encoded = le_device.transform([device_id])[0]
        except ValueError:
            d_encoded = 0

        sample = [[u_encoded, d_encoded, amount, failed_logins, location_change]]
        prediction = model.predict(sample)
        
        if prediction[0] == 1:
            st.error("🚨 [WARNING] تحذير خطير: هذه المعاملة [احتيالية] وتم حجزها بنجاح!")
        else:
            st.success("✅ [SAFE] المعاملة سليمة وطبيعية، مبروك المرور!")
        