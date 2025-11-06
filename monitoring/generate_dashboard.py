import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import os

# --- 1. ตั้งค่า MLflow ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "loan-default-model-cpe393"
MODEL_STAGE = "Production"
# -------------------------

print("Loading data for monitoring...")
# 2. โหลดข้อมูล Reference (Validation set)
try:
    ref_data = pd.read_csv("data/processed/reference_data.csv")
    ref_target = pd.read_csv("data/processed/reference_target.csv")
    ref_data['TARGET'] = ref_target['TARGET']

    # 3. โหลดข้อมูล Current (จำลองโดยใช้ Training set)
    curr_data = pd.read_csv("data/processed/current_data_sim.csv")
    curr_target = pd.read_csv("data/processed/current_target_sim.csv")
    curr_data['TARGET'] = curr_target['TARGET']
except FileNotFoundError:
    print("Error: ไม่พบไฟล์ 'reference_data.csv' หรือ 'current_data_sim.csv'")
    print("โปรดรัน 'src/train.py' ก่อนเพื่อสร้างไฟล์เหล่านี้")
    exit()
    
# 4. โหลดโมเดล Production
print("Loading model from MLflow Registry...")
try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"โปรดตรวจสอบว่าคุณได้ Promote โมเดล '{MODEL_NAME}' ไปยัง Stage '{MODEL_STAGE}' ใน MLflow UI แล้ว")
    exit()

# 5. ใช้โมเดลทำนายผลบนข้อมูลทั้งสองชุด
print("Generating predictions...")
ref_data['prediction'] = model.predict(ref_data.drop('TARGET', axis=1))
curr_data['prediction'] = model.predict(curr_data.drop('TARGET', axis=1))

# 6. สร้าง Evidently AI Report
print("Generating Evidently AI Dashboard...")
drift_report = Report(metrics=[
    DataDriftPreset(),              # ตรวจสอบ Data Drift
    ClassificationPreset()        # ตรวจสอบ Model Performance (Accuracy, Precision, Recall, Drift ฯลฯ)
])

# 7. รัน Report
drift_report.run(
    reference_data=ref_data,       # ข้อมูลอ้างอิง (ตอนเทรน)
    current_data=curr_data,        # ข้อมูลปัจจุบัน (ข้อมูลใหม่)
    column_mapping=None            # Evidently ฉลาดพอที่จะหา 'TARGET' และ 'prediction' เอง
)

# 8. บันทึกเป็น HTML
DASHBOARD_FILE = "monitoring/loan_default_monitoring_dashboard.html"
os.makedirs("monitoring", exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี
drift_report.save_html(DASHBOARD_FILE)

print(f"\nSuccessfully generated dashboard!")
print(f"File saved to: {DASHBOARD_FILE}")