import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import requests # (เพิ่ม)
import zipfile # (เพิ่ม)
import io      # (เพิ่ม)

# 1. กำหนด Features (เหมือนเดิม)
NUMERIC_FEATURES = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'
]
CATEGORICAL_FEATURES = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'
]

# 2. แก้ไขฟังก์ชัน load_and_preprocess_data (อัปเดต)
def load_and_preprocess_data(data_url):
    """
    โหลดข้อมูลจาก URL (bank.zip) และอ่านไฟล์ 'bank-full.csv' ที่อยู่ข้างใน
    """
    try:
        # 1. ดาวน์โหลด .zip file จาก URL
        r = requests.get(data_url)
        r.raise_for_status() # เช็กว่าดาวน์โหลดสำเร็จ
        
        # 2. เปิด .zip file จากใน memory
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # 3. อ่านไฟล์ 'bank-full.csv' ที่อยู่ข้างใน zip
        with z.open('bank-full.csv') as f:
            df = pd.read_csv(f, sep=';') # ระบุ ; เป็นตัวคั่น

    except Exception as e:
        print(f"Error loading data from URL: {e}")
        return None, None, None, None, None

    # 3. แปลง Target (y) (เหมือนเดิม)
    df['TARGET'] = df['y'].map({'yes': 1, 'no': 0})
    
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df_selected = df[all_features + ['TARGET']]

    # 4. แบ่งข้อมูล (เหมือนเดิม)
    X = df_selected.drop('TARGET', axis=1)
    y = df_selected['TARGET']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. สร้าง Preprocessing Pipelines (เหมือนเดิม)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 6. รวม Pipelines (เหมือนเดิม)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    print("Preprocessing pipeline created successfully for Bank dataset.")
    
    return X_train, X_val, y_train, y_val, preprocessor