import pandas as pd
import numpy as np
import os

def generate_transactions(num_samples=1000):
    np.random.seed(42)
    
    # زيادة نسبة الاحتيال إلى 10% حتى يتعلّم الموديل صح
    is_fraud = np.random.choice([0, 1], size=num_samples, p=[0.90, 0.10])
    
    amounts = []
    failed_logins = []
    location_changes = []
    user_ids = [f"user_{np.random.randint(1, 100)}" for _ in range(num_samples)]
    device_ids = [f"device_{np.random.randint(1, 150)}" for _ in range(num_samples)]
    
    for fraud in is_fraud:
        if fraud == 1:
            # معاملات الاحتيال: مبالغ عالية + محاولات دخول فاشلة متكررة + تغيير موقع
            amounts.append(round(np.random.uniform(500, 5000), 2))
            failed_logins.append(np.random.randint(2, 6))
            location_changes.append(np.random.choice([0, 1], p=[0.2, 0.8]))
        else:
            # المعاملات الطبيعية: مبالغ اعتيادية + دخول سليم
            amounts.append(round(np.random.uniform(10, 300), 2))
            failed_logins.append(np.random.choice([0, 1], p=[0.9, 0.1]))
            location_changes.append(np.random.choice([0, 1], p=[0.85, 0.15]))
            
    df = pd.DataFrame({
        'user_id': user_ids,
        'device_id': device_ids,
        'amount': amounts,
        'failed_logins': failed_logins,
        'location_change': location_changes,
        'is_fraud': is_fraud
    })
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'transactions.csv')
    
    df.to_csv(file_path, index=False)
    print(f"[+] Successfully generated {num_samples} transactions ({df['is_fraud'].sum()} fraud cases) at: {file_path}")

if __name__ == "__main__":
    generate_transactions()
    