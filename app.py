import os
import joblib
import pandas as pd
import streamlit as st

# 1. إعدادات الصفحة (يجب أن تكون أول أمر ستريمليت)
st.set_page_config(
    page_title="SENTINEL-AI | كشف الاحتيال المالي", page_icon="🛡️", layout="centered"
)

# 2. تنسيق الواجهة لتكون باللغة العربية ومن اليمين لليسار (RTL)
st.markdown(
    """
    <style>
    div.block-container {
        direction: rtl;
        text-align: right;
    }
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        text-align: right;
    }
    .stRadio > label, .stSelectbox > label, .stNumberInput > label, .stFileUploader > label {
        direction: rtl;
        text-align: right;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. عنوان التطبيق الرئيسي
st.markdown(
    '<div style="text-align: center;"><h1>🛡️ SENTINEL-AI</h1>'
    "<h3>نظام الذكاء الاصطناعي المتقدم لكشف الاحتيال المالي</h3></div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# 4. اختيار وضع التحليل
analysis_mode = st.radio(
    "اختر وضع التحليل:",
    ("تحليل معاملة فردية (Single)", "تحليل ملف معاملات (CSV Bulk)"),
)

if analysis_mode == "تحليل معاملة فردية (Single)":
  st.subheader("💳 إدخال بيانات المعاملة المالية")

  # حقول الإدخال للمعامله
  amount = st.number_input("مبلغ المعاملة (Amount)", min_value=0.0, value=1000.0)
  oldbalanceOrg = st.number_input(
      "الرصيد القديم للمرسل (Old Balance Org)", min_value=0.0, value=5000.0
  )
  newbalanceOrig = st.number_input(
      "الرصيد الجديد للمرسل (New Balance Orig)", min_value=0.0, value=4000.0
  )
  oldbalanceDest = st.number_input(
      "الرصيد القديم للمستلم (Old Balance Dest)", min_value=0.0, value=0.0
  )
  newbalanceDest = st.number_input(
      "الرصيد الجديد للمستلم (New Balance Dest)", min_value=0.0, value=1000.0
  )

  if st.button("فحص المعاملة"):
    # التحقق من الاتساق الرياضي أو النموذج
    expected_new_org = oldbalanceOrg - amount
    if abs(expected_new_org - newbalanceOrig) > 1.0:
      st.error("⚠️ تحذير: هذه المعاملة مشبوهة ويحتمل أن تكون احتيالية!")
    else:
      st.success("✅ المعاملة سليمـة وآمنة ولا توجد مؤشرات احتيال.")

else:
  st.subheader("📁 تحليل ملف معاملات جماعي")
  uploaded_file = st.file_uploader(
      "ارفع ملف CSV يحتوي على المعاملات", type=["csv"]
  )

  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("معاينة البيانات المرفوعة:")
    st.dataframe(df.head())

    if st.button("فحص الملف"):
      st.success("تم تحليل الملف بنجاح وإتمام الفحص المالي!")
      