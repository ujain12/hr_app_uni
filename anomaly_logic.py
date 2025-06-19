import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Rank hierarchy for numerical mapping
RANK_HIERARCHY = {
    'PVT': 1, 'PV2': 2, 'PFC': 3, 'SPC': 4, 'CPL': 5, 'SGT': 6, 'SSG': 7, 'SFC': 8, 'MSG': 9,
    '1SG': 10, 'SGM': 11, 'CSM': 12, 'SMA': 13, 'WO1': 14, 'CW2': 15, 'CW3': 16, 'CW4': 17, 'CW5': 18,
    '2LT': 19, '1LT': 20, 'CPT': 21, 'MAJ': 22, 'LTC': 23, 'COL': 24, 'BG': 25, 'MG': 26, 'LTG': 27, 'GEN': 28
}

SENIORITY_GROUPS = {
    'Enlisted': {'PVT', 'PV2', 'PFC', 'SPC', 'CPL'},
    'NCO': {'SGT', 'SSG', 'SFC', 'MSG', '1SG', 'SGM', 'CSM', 'SMA'},
    'Warrant': {'WO1', 'CW2', 'CW3', 'CW4', 'CW5'},
    'Officer': {'2LT', '1LT', 'CPT', 'MAJ', 'LTC', 'COL', 'BG', 'MG', 'LTG', 'GEN'}
}

def map_rank_level(rank):
    return RANK_HIERARCHY.get(str(rank).strip().upper(), np.nan)

def map_seniority(rank):
    for group, ranks in SENIORITY_GROUPS.items():
        if str(rank).upper() in ranks:
            return group
    return 'Unknown'
def feature_engineering(df):
    df = df.copy()

    # Convert to datetime
    df['DateOfEntryService'] = pd.to_datetime(df['DateOfEntryService'], errors='coerce')
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')

    # Numeric conversions
    df['BasePay'] = pd.to_numeric(df['BasePay'], errors='coerce')
    df['Bonus'] = pd.to_numeric(df['Bonus'], errors='coerce')

    # Years of service
    df['YearsOfService'] = pd.Timestamp.today().year - df['DateOfEntryService'].dt.year

    # Derived features
    df['RankLevel'] = df['Rank'].apply(map_rank_level)
    df['SeniorityGroup'] = df['Rank'].apply(map_seniority)
    df['BasePayPerYear'] = df['BasePay'] / df['YearsOfService'].replace(0, np.nan)
    df['BonusRatio'] = df['Bonus'] / df['BasePay'].replace(0, np.nan)
    df['PromotionRate'] = df['RankLevel'] / df['YearsOfService'].replace(0, np.nan)
    df['DutyStatusFlag'] = (df['DutyStatus'].str.lower() == 'active').astype(int)

    # Binning for entry year
    df['ServiceEra'] = pd.cut(df['DateOfEntryService'].dt.year,
                              bins=[0, 1999, 2010, 2020, 2100],
                              labels=['Pre-2000', '2000-2010', '2010-2020', 'Post-2020'])

    return df


def preprocess_data(df):
    df = feature_engineering(df)
    id_cols = [col for col in df.columns if any(x in col.lower() for x in ['id', 'ssn', 'name', 'email', 'address', 'zip', 'city', 'street'])]
    df_cleaned = df.drop(columns=id_cols, errors='ignore')

    num_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X = preprocessor.fit_transform(df_cleaned)
    return df, X, num_cols + cat_cols

def run_model(df, X, method='isolation', contamination=0.05):
    if method == 'svm':
        model = OneClassSVM(kernel='rbf', gamma='scale', nu=contamination)
        preds = model.fit_predict(X)
    elif method == 'lof':
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        preds = model.fit_predict(X)
    else:
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        preds = model.fit_predict(X)

    df['is_anomaly'] = preds == -1
    return df

def get_anomaly_percentage(df):
    return round(100 * df['is_anomaly'].mean(), 2)

def get_featured_anomalies(df):
    return df[df['is_anomaly'] == True].copy()
