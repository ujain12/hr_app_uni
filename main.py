import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from anomaly_logic import preprocess_data, run_model, get_anomaly_percentage, get_featured_anomalies
from model_selector import evaluate_models
from report_generator import generate_anomaly_report

# ---------- Data Cleaning Functions ----------

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
    df["DateOfBirth"] = pd.to_datetime(df["DateOfBirth"], errors="coerce")
    df["DateOfEntryService"] = pd.to_datetime(df["DateOfEntryService"], errors="coerce")
    df["BasePay"] = pd.to_numeric(df["BasePay"], errors="coerce")
    df["Bonus"] = pd.to_numeric(df["Bonus"], errors="coerce")
    df["YearsOfService"] = (df["DateOfEntryService"] - df["DateOfBirth"]).dt.days / 365.25
    cols = ["BasePay", "Bonus", "YearsOfService"]
    df_numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    df_model = df_numeric.dropna()
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
    for col in dedup_columns:
        if col not in df.columns:
            return df
    return df.drop_duplicates(subset=dedup_columns)

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Unified HR Anomaly App", layout="wide")
st.title("🧹📊 Unified HR Data Cleaner + Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("📤 Upload your HR CSV file", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Cleaning Summary Tracking
    initial_count = len(df_raw)
    df_trimmed = rule_based(df_raw.copy())
    duplicates_removed = initial_count - len(deduplicate_df(df_trimmed))
    invalid_zips = 0
    if "ZipCode" in df_raw.columns:
        invalid_zips = df_raw["ZipCode"].astype(str).apply(lambda x: not re.fullmatch(r'\d{5}', x)).sum()
    missing_values = df_raw.isnull().sum().sum()

    with st.spinner("🧼 Cleaning data..."):
        df_cleaned = rule_based(df_raw.copy())
        df_cleaned = ml_cleaning(df_cleaned)
        df_cleaned = deduplicate_df(df_cleaned)
        st.success("✅ Data cleaning complete.")

    st.subheader("🧹 Data Cleaning Summary")
    st.write(f"• 📉 **Duplicates removed:** {duplicates_removed}")
    st.write(f"• 🚫 **Invalid ZIP codes corrected:** {invalid_zips}")
    st.write(f"• 🕳️ **Total missing values found:** {missing_values}")

    with st.spinner("🤖 Running anomaly detection..."):
        df_processed, X, features = preprocess_data(df_cleaned)
        best_model, scores = evaluate_models(X, features)
        df_results = run_model(df_cleaned, X, method=best_model)
        anomaly_pct = get_anomaly_percentage(df_results)
        anomalies_df = get_featured_anomalies(df_results)

        generate_anomaly_report(anomalies_df)
        st.success("✅ Anomaly detection complete!")

    st.subheader("📈 Anomaly Detection Summary")
    st.metric("🔍 % Anomalies Detected", f"{anomaly_pct:.2f}%")
    st.write(anomalies_df)

    with open("output/anomaly_report.pdf", "rb") as f:
        st.download_button("📥 Download Anomaly Report", f, file_name="anomaly_report.pdf", mime="application/pdf")
