# CPE393 MLOps Final Project: Loan Default Prediction

โปรเจกต์นี้สร้าง End-to-End MLOps Pipeline สำหรับการทำนายการผิดนัดชำระหนี้ (Loan Default Prediction) โดยใช้ข้อมูลจาก [Home Credit Default Risk Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk).

## 🚀 MLOps Tools Used
* **Experiment Tracking:** MLflow
* **Containerization:** Docker
* **Deployment:** FastAPI
* **Automation (CI/CD):** GitHub Actions
* **Monitoring:** Evidently AI
* **Data Handling:** Kaggle API

## 🏗️ Project Structure
loan-default-mlops/ ├── .github/workflows/ │ └── training.yml # (GitHub Actions สำหรับเทรนโมเดลอัตโนมัติ) ├── api/ │ ├── main.py # (FastAPI script) │ ├── Dockerfile # (Dockerfile สำหรับ Deployment) │ └── requirements.txt # (Dependencies สำหรับ API) ├── data/ │ ├── raw/ # (เก็บ application_train.csv) │ └── processed/ # (เก็บไฟล์ที่ train.py สร้าง) ├── monitoring/ │ ├── generate_dashboard.py # (สคริปต์สร้าง Evidently AI Dashboard) │ └── requirements.txt ├── notebooks/ │ └── 01-eda-and-bias.ipynb # (EDA และวิเคราะห์ Bias) ├── src/ │ ├── preprocess.py # (สคริปต์เตรียมข้อมูล) │ ├── train.py # (สคริปต์เทรนโมเดล 3 แบบและ Track ด้วย MLflow) │ └── requirements.txt └── README.md
## ⚙️ How to Run

### 1. Setup MLflow
1.  ติดตั้ง MLflow: `pip install mlflow`
2.  รัน MLflow UI Server: `mlflow ui --host 0.0.0.0 --port 5000`
3.  ตรวจสอบว่า URI ใน `src/train.py`, `api/main.py`, และ `monitoring/generate_dashboard.py` ตรงกับ Server ของคุณ (เช่น `http://127.0.0.1:5000`)

### 2. Setup Kaggle API (Local)
1.  ไปที่เว็บ Kaggle > Your Account > Settings > กด "Create New API Token" เพื่อดาวน์โหลด `kaggle.json`
2.  ย้ายไฟล์ `kaggle.json` ไปไว้ที่ `~/.kaggle/kaggle.json` (บน Mac/Linux) หรือ `C:\Users\<YourUser>\.kaggle\kaggle.json` (บน Windows)

### 3. Training
1.  ติดตั้ง Dependencies: `pip install -r src/requirements.txt`
2.  รันสคริปต์เทรน (สคริปต์จะดาวน์โหลดข้อมูลจาก Kaggle ให้อัตโนมัติ):
    ```bash
    python src/train.py
    ```
3.  ไปที่ MLflow UI (`http://127.0.0.1:5000`) > Experiments > `CPE393-Loan-Default` เพื่อดูผล
4.  ไปที่ Models > `loan-default-model-cpe393` และ Promote โมเดลเวอร์ชันที่ดีที่สุดไปเป็น **"Production"**

### 4. API Deployment (with Docker)
1.  Build Docker image: `docker build -t loan-api-cpe393 .` (รันในโฟลเดอร์ `api/`)
2.  Run Docker container: `docker run -p 8000:80 -e MLFLOW_TRACKING_URI="http://<YOUR_HOST_IP>:5000" loan-api-cpe393`
    * (สำคัญ: ใช้ `http://host.docker.internal:5000` ถ้า Docker รันบน Mac/Windows, หรือ IP ของเครื่อง Host ถ้าเป็น Linux)
3.  เปิด Browser ไปที่ `http://127.0.0.1:8000/docs` เพื่อดู API Docs

### 5. Monitoring
1.  ติดตั้ง Dependencies: `pip install -r monitoring/requirements.txt`
2.  รันสคริปต์สร้าง Dashboard (หลังจากรัน `train.py` แล้ว):
    ```bash
    python monitoring/generate_dashboard.py
    ```
3.  เปิดไฟล์ `monitoring/loan_default_monitoring_dashboard.html` เพื่อดูผล

### 6. Automation (GitHub Actions)
1. Push โค้ดนี้ขึ้น GitHub Repository
2. ไปที่ Settings > Secrets and variables > Actions และตั้งค่า Secrets 4 ตัว:
    * `MLFLOW_TRACKING_URI`
    * `MLFLOW_USERNAME` (ถ้ามี)
    * `MLFLOW_PASSWORD` (ถ้ามี)
    * `KAGGLE_JSON`: (คัดลอก "เนื้อหา" ทั้งหมดในไฟล์ `kaggle.json` มาวาง)
3. ไปที่แท็บ "Actions" ใน Repo ของคุณ, เลือก "CPE393-Model-Retraining" และกด "Run workflow" เพื่อทดสอบ