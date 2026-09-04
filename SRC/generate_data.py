import os
import pandas as pd
import numpy as np

def generate_realistic_data():
    print("[*] Generating realistic synthetic fraud dataset...")
    np.random.seed(42)
    n_samples = 1000
    
    # توليد المعاملات الطبيعية (Normal Transactions)
    amount = np.random.exponential(scale=150, size=n_samples)
    oldbalanceOrg = np.random.uniform(1000, 50000, size=n_samples)
    newbalanceDest = oldbalanceOrg + amount + np.random.normal(0, 5, size=n_samples)
    
    # مصفوفة الهدف (الافتراضي كلها طبيعية 0)
    is_fraud = np.zeros(n_samples, dtype=int)
    
    # إضافة قاعدة منطقية للاحتيال (Fraud Pattern):
    # جعل 5% من المعاملات احتيالية تتميز بمبلغ ضخم جداً ورصيد غير متطابق
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    
    for idx in fraud_indices:
        amount[idx] = np.random.uniform(8000, 25000)  # مبلغ ضخم مشبوه
        oldbalanceOrg[idx] = np.random.uniform(0, 200)   # رصيد الحساب شبه فارغ
        newbalanceDest[idx] = oldbalanceOrg[idx] - amount[idx] # تلاعب بالرصيد
        is_fraud[idx] = 1
        
    # تجميع البيانات في جدول Pandas DataFrame
    df = pd.DataFrame({
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceDest': newbalanceDest,
        'is_fraud': is_fraud
    })
    
    # التأكد من وجود مجلد DATA وحفظ الملف فيه
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'DATA'), exist_ok=True)
    file_path = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'transactions.csv')
    df.to_csv(file_path, index=False)
    print(f"[*] New realistic dataset generated successfully at: {file_path}")

if __name__ == "__main__":
    generate_realistic_data()