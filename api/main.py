import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. ตั้งค่า MLflow ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# -------------------------

# 2. [อัปเดต] สร้าง Pydantic Model สำหรับ "Bank Marketing"
class BankCustomer(BaseModel):
    # Numeric Features
    age: Optional[int] = Field(None)
    balance: Optional[int] = Field(None)
    day: Optional[int] = Field(None)
    duration: Optional[int] = Field(None)
    campaign: Optional[int] = Field(None)
    pdays: Optional[int] = Field(None)
    previous: Optional[int] = Field(None)
    
    # Categorical Features
    job: Optional[str] = Field(None)
    marital: Optional[str] = Field(None)
    education: Optional[str] = Field(None)
    default: Optional[str] = Field(None)
    housing: Optional[str] = Field(None)
    loan: Optional[str] = Field(None)
    contact: Optional[str] = Field(None)
    month: Optional[str] = Field(None)
    poutcome: Optional[str] = Field(None)

    # Pydantic v2 Example
    class Config:
        json_schema_extra = {
            "example": {
                "age": 42,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 2000,
                "housing": "yes",
                "loan": "no",
                "contact": "unknown",
                "day": 5,
                "month": "may",
                "duration": 120,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        }

# 3. โหลดโมเดล (ใช้ Aliases ตามที่เราคุยกัน)
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for CPE393 MLOps Final Project"
)

model = None
MODEL_NAME = "bank-marketing-model-cpe393"
MODEL_ALIAS = "production" # <-- [สำคัญ] ใช้ Alias

@app.on_event("startup")
def load_model():
    global model
    try:
        # [สำคัญ] เปลี่ยน / เป็น @ เพื่อเรียกใช้ Alias
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}" 
        
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model '{MODEL_NAME}' alias '{MODEL_ALIAS}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        pass

@app.get("/")
def read_root():
    return {"message": f"Welcome to Bank Marketing Prediction API. Model '{MODEL_NAME}' is {'loaded' if model else 'NOT loaded'}."}

@app.post("/predict")
def predict_default(customer: BankCustomer): # <-- [อัปเดต] เปลี่ยนชื่อ Input
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check MLflow connection or if model has '@Production' alias.")
        
    try:
        # 4. แปลง Pydantic model เป็น DataFrame
        input_data = customer.model_dump()
        input_df = pd.DataFrame([input_data])
        
        # 5. ทำนายผล
        prediction = model.predict(input_df)
        result = int(prediction[0])

        return {
            "prediction": result,
            "interpretation": "1 means 'Will Subscribe', 0 means 'Will Not Subscribe'"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")