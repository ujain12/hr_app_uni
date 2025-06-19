import pandas as pd 
import re

#data validation
#load data and check cols
df = pd.read_csv("synthetic_ipps_a_employees_noisy.csv")
print("PREVIEW:")
print(df.head())
print("\nCOLUMN TYPES:")
print(df.dtypes)
print("\nROWS X COLUMNS")
print(df.shape)
print("\nCOLUMN NAMES")
print(df.columns)

#check missing values
null= df.map(pd.isnull)
nullCount=null.sum()
print("\nNull values per column:")
print(nullCount)

#check emp id format (begins with EMP)
emp= (~df['EmployeeID'].str.startswith('EMP')).sum()
print("\nIncorrectly formatted Employee ID cells: ", emp)

#check ssn pattern data (XXX-XX-XXXX)
ssn_pattern = re.compile(r'^\d{3}-\d{2}-\d{4}$')
def is_valid_ssn(ssn):
    return bool(ssn_pattern.match(ssn))
ssn = df['SSN_Synthetic'].apply(lambda x: not is_valid_ssn(x)).sum()
print("Incorrectly formatted SSNs:", ssn)

#check fullname pattern (contains first and last name)
name = df.apply(
    lambda row: row['FirstName'].strip() not in row['FullName'] or row['LastName'].strip() not in row['FullName'],
    axis=1
).sum()
print("Incorrectly formatted fullnames:", name)

#check state pattern (2 letters)
state_pattern = re.compile(r'^[A-Z]{2}$')
state = df['State'].apply(lambda x: not bool(state_pattern.fullmatch(str(x).strip()))).sum()
print("Incorrectly formatted States:", state)

#check zip code pattern (5 numbers)
zip_pattern = re.compile(r'^\d{5}$')
zipcode = df['ZipCode'].apply(lambda x: not bool(zip_pattern.fullmatch(str(x).strip()))).sum()
print("Incorrectly formatted zip codes:", zipcode)

#check date of entry service > date of birth
df["DateOfBirth"] = pd.to_datetime(df["DateOfBirth"], format='mixed')
df["DateOfEntryService"] = pd.to_datetime(df["DateOfEntryService"], format='mixed')
df["compare"] = df["DateOfEntryService"] > df["DateOfBirth"]
date = (~df["compare"]).sum()
print("Incorrectly formatted dates of birth/entry:", date)

#check email column for @ and .
email = df["Email"].apply(lambda x: "@" not in str(x) or "." not in str(x)).sum()
print("Incorrectly formatted emails:", email)

#check ranks (3 characters long)
rank = df["Rank"].apply(lambda x: len(str(x).strip()) != 3).sum()
print("Incorrectly formatted ranks:", rank)

#check duty status (letters only)
duty = (~df["DutyStatus"].str.strip().str.isalpha()).sum()
print("Incorrectly formatted duty statuses:", duty)

#check base pay (nonnegative number)
basepay_numeric = pd.to_numeric(df["BasePay"], errors="coerce")
basepay = basepay_numeric.isna() | (basepay_numeric < 0)
basepay_count = basepay.sum()
print("Incorrectly formatted basepays (non-numeric or negative):", basepay_count)

#check bonus (nonnegative number)
bonus_numeric = pd.to_numeric(df["Bonus"], errors="coerce")
bonus = bonus_numeric.isna() | (bonus_numeric < 0)
bonus_count = bonus.sum()
print("Incorrectly formatted bonuses (non-numeric or negative):", bonus_count)

#how many anomalys
anomaly_count = df["IsAnomaly"].sum()
print("Number of anomalies:", anomaly_count)


#rule based flagging
def rule_based(df):

    #data cleansing
    #remove whitespace
    for col in ["FirstName","LastName","Street","City","State","Email","ZipCode"]:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop("compare", axis=1)

    #flag improperly formatted (non 5-digit) zip codes as "invalid"
    zip_pattern = r'^\d{5}$'
    mask = ~df["ZipCode"].astype(str).str.match(zip_pattern)
    df.loc[mask, "ZipCode"] = "invalid"

    #flag improperly formatted bonuses (non-positive) as "invalid"
    df["Bonus"] = df["Bonus"].astype(str)  
    bonus_num = pd.to_numeric(df["Bonus"], errors="coerce")
    df.loc[(bonus_num < 0) | (bonus_num.isna()), "Bonus"] = "invalid"
    return df


#flagging using isolation forest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def ml_based(df):
    # finding anomalies in pay and bonus based on years of service
    df["DateOfBirth"] = pd.to_datetime(df["DateOfBirth"], errors="coerce")
    df["DateOfEntryService"]  = pd.to_datetime(df["DateOfEntryService"], errors="coerce")
    df["YearsOfService"] = ((df["DateOfEntryService"] - df["DateOfBirth"]).dt.days/ 365.25)
    variables = ["BasePay", "Bonus", "YearsOfService"]
    df_input = df[variables].apply(pd.to_numeric, errors="coerce")

    # drop missing vals
    df_model = df_input.dropna()
    idx_model = df_model.index

    # standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)

    # fit isolation forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_scaled)

    # predict, 0 = normal, 1 = anomaly
    pred = iso.predict(X_scaled)
    df["IsAnomaly"] = 0
    df.loc[idx_model, "IsAnomaly"] = (pred == -1).astype(int)
    df["AnomalyType"] = "Normal"
    df.loc[df["IsAnomaly"] == 1, "AnomalyType"] = "IsolationForest"

    print(df[["EmployeeID","BasePay","Bonus","YearsOfService","IsAnomaly"]].head(15))
    df.to_csv("payroll_with_anomalies.csv", index=False)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def missing_values(df):

    #generate 5% missing values in FullName and Email
    original = df[["FullName", "Email"]].copy()
    df_missing = df.copy()           
    rng = np.random.default_rng(seed=42)

    mask_full  = rng.choice([True, False], size=len(df_missing), p=[0.05,0.95])
    mask_email = rng.choice([True, False], size=len(df_missing), p=[0.05,0.95])
    df_missing.loc[mask_full,  "FullName"] = np.nan
    df_missing.loc[mask_email, "Email"]    = np.nan

    # fill missing full names by first name + last Name
    df_missing["FullName_gen"] = df_missing["FullName"].fillna(
        df_missing["FirstName"].str.strip() + " " + df_missing["LastName"].str.strip()
    )

    # KNN‑fill Email
    train = df_missing[df_missing["Email"].notna()]
    test  = df_missing[df_missing["Email"].isna()]
    le_fn = LabelEncoder().fit(train["FirstName"])
    le_ln = LabelEncoder().fit(train["LastName"])

    X_train = pd.DataFrame({
        "fn": le_fn.transform(train["FirstName"]),
        "ln": le_ln.transform(train["LastName"])
    })
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, train["Email"])

    #test email generation
    X_test = pd.DataFrame({
        "fn": le_fn.transform(test["FirstName"]),
        "ln": le_ln.transform(test["LastName"])
    })
    df_missing.loc[test.index, "Email_gen"] = knn.predict(X_test)
    df_missing["Email_gen"].fillna(df_missing["Email"], inplace=True)

    # find success rates compared to original
    fullname_success = (df_missing.loc[mask_full, "FullName_gen"]== original.loc[mask_full, "FullName"]) .mean() * 100
    email_success = (df_missing.loc[mask_email, "Email_gen"] == original.loc[mask_email, "Email"]).mean()* 100
    print(f"FullName fill success: {fullname_success:.2f}%")
    print(f"Email    fill success: {email_success:.2f}%")

    # Mark which rows had a missing email
    df_missing["missing_email"] = mask_email.astype(int)

    # flag incorrectly generated emails under anomaly type KNN
    mask_wrong_email = mask_email & (df_missing["Email_gen"] != original["Email"])
    df_missing.loc[mask_wrong_email, "IsAnomaly"]    = 1
    df_missing.loc[mask_wrong_email, "AnomalyType"]  = "KNN"

    df_missing.to_csv("payroll_missing_vals.csv", index=False)

df=rule_based(df)
ml_based(df)
missing_values(df)
