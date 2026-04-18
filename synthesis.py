import numpy as np
import pandas as pd

def generate_welfare_dataset(n=3000, program="SNAP"):
    race      = np.random.choice([0,1,2,3], n, p=[0.60,0.15,0.18,0.07])
    gender    = np.random.choice([0,1], n, p=[0.48,0.52])
    age       = np.random.randint(18, 65, n)
    income    = np.random.normal(28000, 12000, n).clip(0)
    income   -= (race==1)*3000 + (race==2)*2000
    hh_size   = np.random.poisson(2.8, n).clip(1,8).astype(int)
    employed  = (income > 20000).astype(int)
    disability= np.random.binomial(1, 0.12, n)
    prior_ben = np.random.binomial(1, 0.35, n)

    fpl_threshold = 16000 + hh_size * 4800
    truly_eligible = ((income < fpl_threshold) | (disability==1)).astype(int)

    bias_noise = (race==1)*0.12 + (race==2)*0.08 + (gender==1)*0.04
    noise      = np.random.normal(0, 0.1, n)
    approval_prob = (0.60 * truly_eligible - bias_noise + noise).clip(0,1)
    historical_approval = (approval_prob > 0.5).astype(int)

    df = pd.DataFrame({
        "race": race, "gender": gender, "age": age,
        "income": income.astype(int), "hh_size": hh_size,
        "employed": employed, "disability": disability,
        "prior_benefit": prior_ben,
        "truly_eligible": truly_eligible,
        "historical_approved": historical_approval
    })
    return df