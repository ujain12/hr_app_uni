import pandas as pd


def deduplicate_csv(input_csv, output_csv, deduplication_columns):
    df = pd.read_csv(input_csv)

    deduplicated_df = df.drop_duplicates(subset=deduplication_columns)

    deduplicated_df.to_csv(output_csv, index=False)

    print(f"Deduplicated CSV saved to {output_csv}.")
    print(f"Original rows: {len(df)}, Deduplicated rows: {len(deduplicated_df)}")


if __name__ == "__main__":
    input_csv = "synthetic_ipps_a_employees_noisy.csv"
    output_csv = "synthetic_ipps_a_employees_deduplicated.csv"
    deduplication_columns = ["EmployeeID", "SSN_Synthetic", "FullName"]

    deduplicate_csv(input_csv, output_csv, deduplication_columns)
