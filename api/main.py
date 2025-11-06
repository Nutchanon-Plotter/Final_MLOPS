import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. ตั้งค่า MLflow ---
# (สำคัญ: ต้องชี้ไปที่ Server เดียวกับตอนเทรน)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# -------------------------

# 2. สร้าง Pydantic Model สำหรับ Input (ต้องตรงกับ Features ที่เทรน)
# ใช้ Optional[...] = Field(None) เพื่ออนุญาตให้เป็นค่าว่าง (null)
class LoanApplication(BaseModel):
    AMT_INCOME_TOTAL: Optional[float] = Field(None)
    AMT_CREDIT: Optional[float] = Field(None)
    AMT_ANNUITY: Optional[float] = Field(None)
    DAYS_BIRTH: Optional[int] = Field(None)
    EXT_SOURCE_1: Optional[float] = Field(None)
    EXT_SOURCE_2: Optional[float] = Field(None)
    EXT_SOURCE_3: Optional[float] = Field(None)
    NAME_CONTRACT_TYPE: Optional[str] = Field(None)
    CODE_GENDER: Optional[str] = Field(None)
    FLAG_OWN_CAR: Optional[str] = Field(None)
    FLAG_OWN_REALTY: Optional[str] = Field(None)

    # Pydantic v2 Example
    class Config:
        json_schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "DAYS_BIRTH": -9461,
                "EXT_SOURCE_1": 0.083037,
                "EXT_SOURCE_2": 0.262949,
                "EXT_SOURCE_3": 0.139376,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y"
            }
        }

# 3. โหลดโมเดลจาก MLflow Model Registry
app = FastAPI(
    title="Loan Default Prediction API",
    description="API for CPE393 MLOps Final Project"
)

model = None
MODEL_NAME = "loan-default-model-cpe393"
MODEL_STAGE = "Production" # (คุณต้องไปกด "Promote" ใน MLflow UI ให้เป็น Production ก่อน)

@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model '{MODEL_NAME}' stage '{MODEL_STAGE}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        # ใน Production จริง อาจจะไม่อยากให้ API สตาร์ทถ้าโหลดโมเดลไม่ได้
        # แต่สำหรับโปรเจกต์, เราจะปล่อยให้มันสตาร์ทและแสดง Error ที่ /predict
        pass

@app.get("/")
def read_root():
    return {"message": f"Welcome to Loan Default Prediction API. Model '{MODEL_NAME}' is {'loaded' if model else 'NOT loaded'}."}

@app.post("/predict")
def predict_default(application: LoanApplication):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check MLflow connection or if model exists in 'Production' stage.")
        
    try:
        # 4. แปลง Pydantic model เป็น DataFrame
        # (ต้องใส่ [0] เพราะ model.predict คาดหวัง input ที่เป็น list-like)
        input_data = application.model_dump()
        input_df = pd.DataFrame([input_data])
        
        # 5. ทำนายผล
        # model ที่โหลดมาคือ Pipeline ที่สมบูรณ์ (Preprocessor + Classifier)
        # มันจะจัดการ Impute, Scale, One-Hot และทำนายผลให้เอง
        prediction = model.predict(input_df)
        
        result = int(prediction[0]) # ผลลัพธ์เป็น 0 หรือ 1
        
        return {
            "prediction": result,
            "interpretation": "1 means 'Likely to Default', 0 means 'Unlikely to Default'"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")