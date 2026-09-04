import streamlit as st
import pandas as pd
import joblib
import os

# 1. إعدادات الصفحة وتصميم الواجهة
st.set_page_config(
    page_title="SENTINEL-AI | كشف الاحتيال المالي",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
    <div style='text-align: center;'>
        <h1>🛡️ SENTINEL-AI</h1>
        <p style='color: gray;'>نظام الذكاء الاصطناعي المتقدم لكشف الاحتيال المالي</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join('MODELS', 'fraud_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("❌ تنبيه: لم يتم العثور على ملف النموذج التدريبي في مجلد MODELS.")
else:
    # الاختيار صار بوسط الشاشة راساً بدون قائمة جانبية!
    app_mode = st.radio(
        "📌 اختر وضع التحليل:",
        ["تحليل معاملة فردية (Single)", "تحليل ملف معاملات CSV (Bulk)"],
        horizontal=True
    )

    st.markdown("---")

    if app_mode == "تحليل معاملة فردية (Single)":
        st.subheader("💳 إدخال بيانات المعاملة المالية")
        
        amount = st.number_input("مبلغ المعاملة (Amount)", min_value=0.0, value=15000.0, step=100.0)
        old_balance = st.number_input("الرصيد السابق للمرسل (Old Balance)", min_value=0.0, value=5000.0, step=100.0)
        new_balance = st.number_input("الرصيد الجديد للمستلم (New Balance)", value=-4850.0, step=100.0)

        if st.button("🔍 تحليل ومراجعة المعاملة"):
            input_data = pd.DataFrame([[amount, old_balance, new_balance]], 
                                      columns=['amount', 'oldbalanceOrg', 'newbalanceDest'])
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100

            st.markdown("### 📄 تقرير نتيجة التحليل")
            
            if prediction == 1:
                st.error(f"🚨 **النتيجة: معاملة مشبوهة (Fraud Detected)**")
                st.warning(f"⚠️ **مستوى الخطورة:** High ({probability:.1f}%)")
                st.markdown("#### 🌸 التفسير التحليلي للقرار:")
                if amount > old_balance:
                    st.markdown("- مبلغ المعاملة يفوق الرصيد السابق المتاح للمرسل.")
                if new_balance < 0:
                    st.markdown("- تضارب حاد في الرصيد الجديد.")
            else:
                st.success(f"✅ **النتيجة: معاملة آمنة وطبيعية (Safe Transaction)**")
                st.info(f"🛡️ **مستوى الأمان عالي** (نسبة الخطورة: {probability:.1f}%)")

    else:
        st.subheader("📁 رفع ملف معاملات جماعي (CSV Bulk Upload)")
        st.markdown("قم برفع ملف بصيغة CSV ليتم فحص المعاملات دفعة واحدة.")
        
        uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"])
        
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write("📋 معاينة البيانات المرفوعة:", df_batch.head())
            
            if st.button("🚀 بدء تحليل الملف الجماعي"):
                try:
                    required_cols = ['amount', 'oldbalanceOrg', 'newbalanceDest']
                    if all(col in df_batch.columns for col in required_cols):
                        X_batch = df_batch[required_cols]
                        preds = model.predict(X_batch)
                        probs = model.predict_proba(X_batch)[:, 1] * 100
                        
                        df_batch['Prediction'] = ['Fraud' if p == 1 else 'Safe' for p in preds]
                        df_batch['Risk Score (%)'] = probs.round(2)
                        
                        st.success("✅ تم الانتهاء من التحليل بنجاح!")
                        st.dataframe(df_batch)
                        
                        csv_result = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 تحميل تقرير النتائج (CSV)",
                            data=csv_result,
                            file_name='sentinel_ai_fraud_report.csv',
                            mime='text/csv',
                        )
                    else:
                        st.error("❌ الملف المرفوع لا يحتوي على الأعمدة المطلوبة بدقة.")
                except Exception as e:
                    st.error(f"حدث خطأ أثناء معالجة الملف: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Project SENTINEL-AI | Developed with 🤍 for Academic Excellence</p>", unsafe_allow_html=True)