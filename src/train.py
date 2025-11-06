import mlflow
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from preprocess import load_and_preprocess_data

# --- 0. [อัปเดต] ส่วนดาวน์โหลดข้อมูลจาก Kaggle API ---
import kaggle
import zipfile
import os

DATA_DIR = "data/raw"
ZIP_FILE = f"{DATA_DIR}/home-credit-default-risk.zip"
CSV_FILE = f"{DATA_DIR}/application_train.csv"
DATA_PATH = CSV_FILE # กำหนด DATA_PATH ไว้ตรงนี้

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(DATA_DIR, exist_ok=True)

# ตรวจสอบว่าไฟล์ CSV ปลายทางมีอยู่หรือยัง
if not os.path.exists(CSV_FILE):
    print("Data not found. Downloading from Kaggle...")
    try:
        # ดาวน์โหลดไฟล์ (ต้องมี kaggle.json)
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            "home-credit-default-risk", # ชื่อ Competition
            path=DATA_DIR,
            quiet=False
        )
        
        # แตกไฟล์ .zip (เฉพาะไฟล์ที่ต้องการ)
        print(f"Extracting {ZIP_FILE}...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extract("application_train.csv", DATA_DIR)
        
        # (ลบไฟล์ zip ทิ้ง)
        os.remove(ZIP_FILE)
        print("Download and extraction complete.")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("โปรดตรวจสอบว่าคุณวางไฟล์ 'kaggle.json' ถูกที่แล้ว (เช่น ~/.kaggle/kaggle.json)")
        exit()
else:
    print("Data already exists. Skipping download.")
# --- จบส่วนอัปเดต ---


# --- 1. ตั้งค่า MLflow ---
# (สำคัญ: แก้เป็น URI ของ Server คุณ)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("CPE393-Loan-Default")
# -------------------------

# 2. โหลดและเตรียมข้อมูล
print("Loading data...")
# (DATA_PATH ถูกกำหนดไว้ด้านบนแล้ว)
X_train, X_val, y_train, y_val, preprocessor = load_and_preprocess_data(DATA_PATH)

if X_train is None:
    exit()

# --- 3. โจทย์บังคับ: เทรนโมเดลอย่างน้อย 3 แบบ ---
models_to_train = {
    "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    "LightGBM": LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.1)
}
# ----------------------------------------------------

best_auc = -1
best_run_id = None
model_registry_name = "loan-default-model-cpe393"

print("Starting model training...")

for model_name, model in models_to_train.items():
    
    # 4. เริ่ม MLflow Run (วัดแยกกัน)
    with mlflow.start_run(run_name=model_name) as run:
        print(f"\nTraining model: {model_name}")
        
        # 5. สร้าง Pipeline ที่รวม Preprocessor และ Model เข้าด้วยกัน
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # 6. เทรนโมเดล
        full_pipeline.fit(X_train, y_train)

        # 7. ประเมินผล
        preds_proba = full_pipeline.predict_proba(X_val)[:, 1]
        preds = full_pipeline.predict(X_val)
        
        auc = roc_auc_score(y_val, preds_proba)
        accuracy = accuracy_score(y_val, preds)

        print(f"  {model_name} ROC-AUC: {auc:.4f}")
        print(f"  {model_name} Accuracy: {accuracy:.4f}")

        # 8. บันทึกลง MLflow (Tracking)
        mlflow.log_param("model_type", model_name)
        
        if model_name == "RandomForest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", accuracy)

        # 9. บันทึก Pipeline ทั้งหมด
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            input_example=X_train.head(5)
        )
        
        mlflow.set_tag("project", "CPE393 Final")

        # 10. เก็บ Run ID ของโมเดลที่ดีที่สุดไว้
        if auc > best_auc:
            best_auc = auc
            best_run_id = run.info.run_id
            
print("\nTraining complete.")
print(f"Best model Run ID: {best_run_id} with ROC-AUC: {best_auc:.4f}")

# 11. ลงทะเบียนโมเดลที่ดีที่สุด
if best_run_id:
    print(f"Registering best model to '{model_registry_name}'...")
    model_uri = f"runs:/{best_run_id}/model"
    
    mlflow.register_model(
        model_uri=model_uri,
        name=model_registry_name
    )
    print("Model registered successfully.")

# 12. บันทึกข้อมูล Validation set ไว้สำหรับ Evidently AI
print("Saving validation data for monitoring...")
os.makedirs("data/processed", exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี
X_val.to_csv("data/processed/reference_data.csv", index=False)
y_val.to_csv("data/processed/reference_target.csv", index=False)
X_train.to_csv("data/processed/current_data_sim.csv", index=False) # จำลองข้อมูลใหม่
y_train.to_csv("data/processed/current_target_sim.csv", index=False) # จำลองข้อมูลใหม่