import os
import numpy as np
import pandas as pd

def generate_patient_data(n_samples=500, random_seed=42):
    """
    Generates a realistic synthetic heart disease dataset of n_samples.
    Features are designed with underlying mathematical correlations to the target
    variable to ensure Scikit-Learn models can learn and evaluate effectively.
    """
    np.random.seed(random_seed)
    
    # 1. Generate core patient features
    age = np.random.randint(29, 80, size=n_samples)
    sex = np.random.binomial(n=1, p=0.65, size=n_samples) # 65% male
    
    # cp: chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal, 3: asymptomatic)
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.3, 0.2, 0.35, 0.15])
    
    # trestbps: resting blood pressure (mmHg)
    trestbps = np.random.normal(130, 18, size=n_samples).astype(int)
    trestbps = np.clip(trestbps, 90, 200) # clip to realistic physiological bounds
    
    # chol: serum cholesterol (mg/dl)
    chol = np.random.normal(240, 45, size=n_samples).astype(int)
    chol = np.clip(chol, 120, 500)
    
    # thalach: maximum heart rate achieved (bpm)
    # Physiological rule: max heart rate tends to decrease with age
    base_max_hr = 220 - age
    thalach = (base_max_hr - np.random.normal(25, 15, size=n_samples)).astype(int)
    thalach = np.clip(thalach, 70, 200)
    
    # 2. Simulate latent risk score to determine the heart disease target (0 or 1)
    # Coefficients matching typical risk factor directions (Logistic sigmoid basis)
    # Lower max heart rate (thalach), higher age, typical chest pain (cp), male sex, higher BP & chol increase risk
    logit = (
        0.04 * (age - 55) +
        0.8 * sex +
        0.6 * cp -
        0.03 * (thalach - 130) +
        0.015 * (trestbps - 130) +
        0.008 * (chol - 240) -
        0.5 # Intercept
    )
    
    # Convert logit to probability using sigmoid function
    prob = 1 / (1 + np.exp(-logit))
    
    # Generate target labels (1 = heart disease, 0 = normal)
    target = np.random.binomial(n=1, p=prob)
    
    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "target": target
    })
    
    # 3. Simulate raw data deficiencies for demonstrating cleaning pipelines
    # Add a few missing values (NaN) in 'chol' and 'trestbps'
    nan_mask_chol = np.random.rand(n_samples) < 0.05 # 5% missing
    nan_mask_bp = np.random.rand(n_samples) < 0.03   # 3% missing
    
    df.loc[nan_mask_chol, "chol"] = np.nan
    df.loc[nan_mask_bp, "trestbps"] = np.nan
    
    # Add a few clear outlier values in resting blood pressure to clean (e.g. 999)
    outlier_indices = np.random.choice(n_samples, size=3, replace=False)
    df.loc[outlier_indices, "trestbps"] = 999.0
    
    return df

def clean_data(df):
    """
    Cleans the raw dataset:
    - Handles missing values via median imputation.
    - Standardizes / winsorizes outliers to reasonable bounds.
    """
    df_clean = df.copy()
    
    print("--- Starting Data Cleaning ---")
    print(f"Initial missing values:\n{df_clean.isnull().sum()}\n")
    
    # 1. Address extreme outliers (e.g., blood pressure = 999) before imputing
    bp_threshold = 250
    extreme_bp_mask = df_clean["trestbps"] > bp_threshold
    print(f"Identified {extreme_bp_mask.sum()} extreme resting blood pressure outliers (> {bp_threshold} mmHg)")
    df_clean.loc[extreme_bp_mask, "trestbps"] = np.nan
    
    # 2. Impute missing values with column medians
    bp_median = df_clean["trestbps"].median()
    chol_median = df_clean["chol"].median()
    
    print(f"Imputing missing 'trestbps' with median: {bp_median} mmHg")
    df_clean["trestbps"] = df_clean["trestbps"].fillna(bp_median)
    
    print(f"Imputing missing 'chol' with median: {chol_median} mg/dl")
    df_clean["chol"] = df_clean["chol"].fillna(chol_median)
    
    # Cast to proper integer types after filling NaNs
    df_clean["trestbps"] = df_clean["trestbps"].astype(int)
    df_clean["chol"] = df_clean["chol"].astype(int)
    
    # 3. Validate ranges
    print("\nVerification of cleaned ranges:")
    print(f"Age range: {df_clean['age'].min()} - {df_clean['age'].max()}")
    print(f"Cholesterol range: {df_clean['chol'].min()} - {df_clean['chol'].max()}")
    print(f"BP range: {df_clean['trestbps'].min()} - {df_clean['trestbps'].max()}")
    print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
    print("--- Data Cleaning Complete ---\n")
    
    return df_clean

if __name__ == "__main__":
    # Ensure folders exist
    os.makedirs("data", exist_ok=True)
    
    # Generate synthetic raw data
    raw_df = generate_patient_data(n_samples=500, random_seed=42)
    raw_df.to_csv("data/raw_patients.csv", index=False)
    print("Saved simulated raw patient dataset to 'data/raw_patients.csv'")
    
    # Clean the dataset
    clean_df = clean_data(raw_df)
    clean_df.to_csv("data/clean_patients.csv", index=False)
    print("Saved clean patient dataset to 'data/clean_patients.csv'")
