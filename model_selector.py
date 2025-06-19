from anomaly_logic import run_model
from sklearn.metrics import f1_score
import numpy as np


def evaluate_models(df, X):
    results = {}

    for method in ['isolation', 'lof', 'svm']:
        try:
            df_temp = run_model(df.copy(), X, method)
            y_true = df_temp.get('IsAnomaly')
            y_pred = df_temp['is_anomaly']

            if y_true is not None and not y_true.isnull().all():
                f1 = f1_score(y_true, y_pred)
            else:
                f1 = df_temp['is_anomaly'].mean()  # Use % anomaly if ground truth not available

            results[method] = f1
        except Exception as e:
            results[method] = -1  # Penalize failures

    best_method = max(results, key=results.get)
    return best_method, results[best_method]
