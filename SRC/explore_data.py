import pandas as pd
import os

def explore_transactions():
    print("[*] Loading and exploring transactions dataset...")
    
    # تحديد مسار ملف البيانات الخام بدقة
    file_path = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'raw', 'transactions.csv')
    
    # قراءة البيانات باستخدام Pandas
    df = pd.read_csv(file_path)
    
    # 1. عرض معلومات عامة عن أبعاد الجدول (الصفوف والأعمدة)
    print(f"\n[📊] Dataset Shape (Rows, Columns): {df.shape}")
    print("\n[📋] Columns in Dataset:")
    print(df.columns.tolist())
    
    # 2. عرض عينة من أول 5 صفوف
    print("\n[👀] First 5 rows of data:")
    print(df.head())
    
    # 3. فحص القيم المفقودة (Missing Values)
    print("\n[🔍] Missing Values per Column:")
    print(df.isnull().sum())
    
    # 4. فحص الصفوف المتكررة (Duplicates)
    print(f"\n[🔄] Duplicate Rows Count: {df.duplicated().sum()}")
    
    # 5. توزيع العمليات (طبيعي مقابل احتيال)
    print("\n[⚖️] Fraud vs Normal Distribution:")
    print(df['is_fraud'].value_counts())

if __name__ == "__main__":
    explore_transactions()
    