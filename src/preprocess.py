import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. กำหนด Features ที่จะใช้ (เลือกมาแค่บางส่วน)
# นี่คือส่วนที่คุณต้องไปทำ Feature Engineering เพิ่มเติม
NUMERIC_FEATURES = [
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'DAYS_BIRTH',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]

CATEGORICAL_FEATURES = [
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY'
]

def load_and_preprocess_data(data_path):
    """
    โหลดข้อมูล, เลือก Features, และสร้าง Preprocessing Pipeline
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: ไม่พบไฟล์ข้อมูลที่ {data_path}")
        print("โปรดดาวน์โหลดข้อมูล 'application_train.csv' จาก Kaggle แล้ววางไว้ใน 'data/raw/'")
        return None, None, None, None, None

    # เลือกเฉพาะ Features ที่กำหนด + TARGET
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    # กรองเอาเฉพาะ GENDER ที่เป็น M หรือ F (ข้อมูลมี XNA)
    df = df[df['CODE_GENDER'].isin(['M', 'F'])]
    
    df_selected = df[all_features + ['TARGET']]

    # 2. แบ่งข้อมูล
    X = df_selected.drop('TARGET', axis=1)
    y = df_selected['TARGET']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. สร้าง Preprocessing Pipelines
    # Pipeline สำหรับ Numeric Features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # เติมค่าว่างด้วย median
        ('scaler', StandardScaler())                 # Scale ข้อมูล
    ])

    # Pipeline สำหรับ Categorical Features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # เติมค่าว่างด้วยค่าที่พบบ่อยสุด
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # ทำ One-Hot Encoding
    ])

    # 4. รวม Pipelines ทั้งหมดด้วย ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    print("Preprocessing pipeline created successfully.")
    
    # ส่งคืนข้อมูลที่ยังไม่ถูก transform
    # เพราะเราจะรวม preprocessor เข้ากับ model ใน 'train.py'
    return X_train, X_val, y_train, y_val, preprocessor