from sqlalchemy import create_engine
import pandas as pd
from synthesis import generate_welfare_dataset

def load_real_data():
    engine = create_engine("sqlite:///fairness.db")
    df = pd.read_sql("SELECT * FROM adult_dataset", engine)
    df = df.fillna("Unknown")
    return df


def load_synthetic_data(n=3000):
    df = generate_welfare_dataset(n)
    return df