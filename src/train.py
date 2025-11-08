import mlflow
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from preprocess import load_and_preprocess_data 
import os

# --- 0. [อัปเดต] ใช้ URL โดยตรงจาก UCI ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
# ----------------------------------------------

# --- 1. ตั้งค่า MLflow ---
MLFLOW_TRACKING_URI = "https://dagshub.com/plotter.natchanon/Final_MLOPS"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("CPE393-Bank-Marketing") # (เปลี่ยนชื่อ Experiment)
# -------------------------

# 2. โหลดและเตรียมข้อมูล
print(f"Loading data from: {DATA_URL}")

X_train, X_val, y_train, y_val, preprocessor = load_and_preprocess_data(DATA_URL) 

if X_train is None:
    print(f"Error: ไม่สามารถโหลดข้อมูลได้จาก {DATA_URL}")
    exit()

# --- 3. โจทย์บังคับ: เทรนโมเดล 3 แบบ ---
models_to_train = {
    "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    "LightGBM": LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.1)
}
# ----------------------------------------------------

best_auc = -1
best_run_id = None
model_registry_name = "bank-marketing-model-cpe393" # (เปลี่ยนชื่อโมเดล)

print("Starting model training...")

# --- นี่คือบรรทัดที่ 47 (for loop) ---
for model_name, model in models_to_train.items():
    
    # --- โค้ดทั้งหมดนี้ต้องย่อหน้า (Indented) ---
    
    # 4. เริ่ม MLflow Run
    with mlflow.start_run(run_name=model_name) as run:
        print(f"\nTraining model: {model_name}")
        
        # 5. สร้าง Pipeline
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

        # 8. บันทึกลง MLflow
        mlflow.log_param("model_type", model_name)
        
        if model_name == "RandomForest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", accuracy)

        # 9. บันทึก Pipeline
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            input_example=X_train.head(5)
        )
        
        mlflow.set_tag("project", "CPE393 Final")

        # 10. เก็บ Run ID ที่ดีที่สุด
        if auc > best_auc:
            best_auc = auc
            best_run_id = run.info.run_id
            
# --- สิ้นสุด Block ของ for loop ---

print("\nTraining complete.")
print(f"Best model Run ID: {best_run_id} with ROC-AUC: {best_auc:.4f}")

# 11. ลงทะเบียนโมเดล
if best_run_id:
    print(f"Registering best model to '{model_registry_name}'...")
    model_uri = f"runs:/{best_run_id}/model"
    
    mlflow.register_model(
        model_uri=model_uri,
        name=model_registry_name
    )
    print("Model registered successfully.")

# 12. บันทึกข้อมูลสำหรับ Evidently AI
print("Saving validation data for monitoring...")
os.makedirs("data/processed", exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี
X_val.to_csv("data/processed/reference_data.csv", index=False)
y_val.to_csv("data/processed/reference_target.csv", index=False)
X_train.to_csv("data/processed/current_data_sim.csv", index=False)
y_train.to_csv("data/processed/current_target_sim.csv", index=False)