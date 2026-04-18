from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter

# 🔥 FORCE load .env properly
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# 🔥 Get DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

print("DB URL:", DATABASE_URL)  # DEBUG FIRST

# ❗ Check before using
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL is not set. Check your .env file")

# 🔥 Create engine
engine = create_engine(DATABASE_URL)

# 🔥 Load dataset
file_path = "adult.csv"

import pandas as pd

df = pd.read_csv(
    "adult.csv",
    header=None,
    sep=",",
    skipinitialspace=True,
    na_values="?",
    engine="python"
)
# Add column names
df.columns = [
    "age","workclass","fnlwgt","education","education_num",
    "marital_status","occupation","relationship","race","sex",
    "capital_gain","capital_loss","hours_per_week","native_country","income"
]

print(df.head())

print("First 5 records:\n", df.head())

# 🔥 Save to database
df.to_sql("adult_dataset", engine, if_exists="replace", index=False)

print("✅ Data saved to database")