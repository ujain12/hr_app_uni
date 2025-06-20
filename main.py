import streamlit as st
import pandas as pd
import numpy as np
import re
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from anomaly_logic import preprocess_data, run_model, get_anomaly_percentage, get_featured_anomalies
from model_selector import evaluate_models
from report_generator import generate_anomaly_report
from datacleaning import rule_based, ml_based, missing_values

# ---------- Page Setup ----------
st.set_page_config(page_title="Army Payroll Cleaning & Anomaly Detection", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Logo + Title ----------
logo_path = "VIZ_Full Logo_Blue.png"
try:
    logo = Image.open(logo_path)
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(logo, width=120)
    with col2:
        st.title("Army Payroll Data Cleaner and Anomaly Detection  ")
except FileNotFoundError:
    st.warning(f"Logo file '{logo_path}' not found. Please place it in the app folder.")

# ---------- Utility Cleaning Functions ----------
def rule_based(df):
    for col in ["FirstName", "LastName", "Street", "City", "State", "Email", "ZipCode"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "compare" in df.columns:
        df = df.drop("compare", axis=1)
    if "ZipCode" in df.columns:
        mask_zip = ~df["ZipCode"].astype(str).str.match(r'^\d{5}$')
        df.loc[mask_zip, "ZipCode"] = "invalid"
    return df

def ml_cleaning(df):
    df["DateOfBirth"] = pd.to_datetime(df.get("DateOfBirth"), errors="coerce")
    df["DateOfEntryService"] = pd.to_datetime(df.get("DateOfEntryService"), errors="coerce")
    df["BasePay"] = pd.to_numeric(df.get("BasePay"), errors="coerce")
    df["Bonus"] = pd.to_numeric(df.get("Bonus"), errors="coerce")
    df["YearsOfService"] = (df["DateOfEntryService"] - df["DateOfBirth"]).dt.days / 365.25
    cols = ["BasePay", "Bonus", "YearsOfService"]
    df_model = df[cols].dropna()
    idx = df_model.index
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(X_scaled)
    df["AnomalyType"] = df.get("AnomalyType", "Normal")
    df.loc[idx[preds == -1], "AnomalyType"] = "IsolationForest"
    return df

def deduplicate_df(df):
    dedup_columns = ["EmployeeID", "SSN_Synthetic", "FullName"]
    return df.drop_duplicates(subset=[col for col in dedup_columns if col in df.columns])

# ---------- App Logic ----------
uploaded_file = st.file_uploader("Upload Army HR/Payroll CSV File", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")

    initial_count = len(df_raw)
    df_trimmed = rule_based(df_raw.copy())
    duplicates_removed = initial_count - len(deduplicate_df(df_trimmed))
    invalid_zips = df_raw["ZipCode"].astype(str).apply(lambda x: not re.fullmatch(r'\d{5}', x)).sum() if "ZipCode" in df_raw.columns else 0
    missing_values_total = df_raw.isnull().sum().sum()

    with st.spinner("Performing data cleaning..."):
        df_cleaned = rule_based(df_raw.copy())
        df_cleaned = ml_cleaning(df_cleaned)
        df_cleaned = deduplicate_df(df_cleaned)
        df_cleaned, email_success, emails_generated = missing_values(df_cleaned)
        st.success("Data cleaning completed successfully.")

    st.subheader("Cleaning Summary")
    st.markdown(f"""
    - **Duplicates removed:** {duplicates_removed}  
    - **Invalid ZIP codes corrected:** {invalid_zips}  
    - **Total missing values handled:** {missing_values_total}  
    - **Emails generated:** {emails_generated}  
    - **Email generation success rate:** {email_success:.2f}%  
    """)

    with st.spinner("Running anomaly detection..."):
        df_processed, X, features = preprocess_data(df_cleaned)
        best_model, scores = evaluate_models(X, features)
        df_results = run_model(df_cleaned, X, method=best_model)
        anomaly_pct = get_anomaly_percentage(df_results)
        anomalies_df = get_featured_anomalies(df_results)
        generate_anomaly_report(anomalies_df)
        st.success("Anomaly detection completed.")

    st.subheader("Anomaly Detection Summary")
    st.metric("Detected Anomalies (%)", f"{anomaly_pct:.2f}%")
    st.markdown(f"**{len(anomalies_df)} rows were detected as anomalies.** Below are the anomaly rows:")
    st.dataframe(anomalies_df)

    with open("output/anomaly_report.pdf", "rb") as f:
        st.download_button("Download Anomaly Report (PDF)", f, file_name="anomaly_report.pdf", mime="application/pdf")

    # Download cleaned CSV (without anomalies)
    df_final_cleaned = df_results[df_results['is_anomaly'] == False].drop(columns=['is_anomaly', 'AnomalyType'], errors='ignore')
    csv = df_final_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cleaned Payroll File (CSV)", csv, file_name="cleaned_payroll_data.csv", mime="text/csv")
