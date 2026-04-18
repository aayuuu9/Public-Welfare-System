from fastapi import FastAPI
from sqlalchemy import create_engine, text
import pandas as pd

# 🔥 Import ONLY functions (not variables)
from data_loader import load_real_data
from ml_pipeline import run_fraud_case_study, fairness_by_model_race, dp_selection_rates

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔹 DATABASE CONFIG
# =========================
DATABASE_URL = "sqlite:///C:/mini_6/welfare_fairml_system/fairness.db"
engine = create_engine(DATABASE_URL)


# =========================
# 🔹 HOME
# =========================
@app.get("/")
def home():
    return {"message": "API Running"}


# =========================
# 🔹 CHECK TABLES
# =========================
@app.get("/check")
def check():
    df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", engine)
    return df.to_dict()


# =========================
# 🔹 GET SAMPLE DATA
# =========================
@app.get("/data")
def get_data():
    try:
        df = load_real_data()
        return df.head(20).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


# =========================
# 🔹 BASIC FAIRNESS (DB LEVEL)
# =========================
@app.get("/fairness")
def fairness():
    try:
        df = pd.read_sql("SELECT race, sex, income FROM adult_dataset", engine)

        df = df.fillna("Unknown")

        result = (
            df.groupby(["race", "sex"])["income"]
            .value_counts(normalize=True)
            .reset_index(name="ratio")
        )

        return result.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


# =========================
# 🔹 DB CONNECTION TEST
# =========================
@app.get("/test-db")
def test_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "DB connected"}
    except Exception as e:
        return {"error": str(e)}


# =========================
# 🔹 FAIRNESS WITH DP
# =========================
@app.get("/fairness-dp")
def fairness_dp():
    return {
        "original": fairness_by_model_race,
        "dp_protected": dp_selection_rates
    }


# =========================
# 🔹 FRAUD CASE STUDY
# =========================
@app.get("/fraud-case-study")
def fraud_case_study():
    return run_fraud_case_study()


# =========================
# 🔹 FULL RESULTS (JSON)
# =========================
@app.get("/results")
def get_results():
    import json
    try:
        with open("results.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}