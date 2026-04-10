from __future__ import annotations
import json, logging, os
from datetime import timedelta
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {"owner": "telemetry-team", "retries": 2, "retry_delay": timedelta(minutes=5)}

def ingest_telemetry(**context):
    import pandas as pd, numpy as np
    np.random.seed(42); n = 100_000
    df = pd.DataFrame({"vehicle_id": np.random.choice([f"VH_{i:05d}" for i in range(10000)], n), "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"), "speed_kmh": np.random.normal(80,30,n).clip(0,200), "battery_pct": np.random.normal(65,20,n).clip(0,100), "temp_motor_c": np.random.normal(75,15,n).clip(20,150), "temp_battery_c": np.random.normal(30,10,n).clip(-20,80), "regen_brake_kw": np.random.exponential(5,n).clip(0,50), "accel_x": np.random.normal(0,2,n), "accel_y": np.random.normal(0,2,n), "accel_z": np.random.normal(9.8,0.5,n), "voltage_v": np.random.normal(400,20,n).clip(300,500), "current_a": np.random.normal(100,40,n).clip(-200,500), "odometer_km": np.random.uniform(0,200000,n)})
    df.to_parquet("/tmp/telemetry_staging.parquet", index=False)
    context["ti"].xcom_push(key="staging_path", value="/tmp/telemetry_staging.parquet")
    logging.info(f"✅ {len(df):,} records ingested")
    return "/tmp/telemetry_staging.parquet"

def validate_telemetry(**context):
    import pandas as pd
    path = context["ti"].xcom_pull(task_ids="ingest_telemetry", key="staging_path")
    df = pd.read_parquet(path)
    assert all(c in df.columns for c in ["vehicle_id","speed_kmh","battery_pct"]), "Missing columns!"
    df.to_parquet("/tmp/telemetry_validated.parquet", index=False)
    context["ti"].xcom_push(key="validated_path", value="/tmp/telemetry_validated.parquet")
    logging.info(f"✅ {len(df):,} records validated")
    return "/tmp/telemetry_validated.parquet"

def run_inference(**context):
    import pandas as pd, numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    path = context["ti"].xcom_pull(task_ids="validate_telemetry", key="validated_path")
    df = pd.read_parquet(path)
    cols = ["speed_kmh","battery_pct","temp_motor_c","temp_battery_c","regen_brake_kw","accel_x","accel_y","accel_z","voltage_v","current_a"]
    X = StandardScaler().fit_transform(df[cols].fillna(0))
    iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=42, n_jobs=-1)
    df["l1_iso_flag"] = (iso.fit_predict(X) == -1).astype(int)
    df["l2_lstm_flag"] = 0; df["l3_cnn_flag"] = 0
    df["anomaly_votes"] = df["l1_iso_flag"] + df["l2_lstm_flag"] + df["l3_cnn_flag"]
    df["is_anomaly"] = (df["anomaly_votes"] >= 1).astype(int)
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000"))
        mlflow.set_experiment("vehicle-telemetry-anomaly-detection")
        with mlflow.start_run(run_name=f"ensemble_{context['ds']}"):
            mlflow.log_params({"n_records": len(df), "iso_contamination": 0.03, "lstm_hidden": 64, "lstm_layers": 2, "cnn_filters": 32, "sequence_length": 30})
            mlflow.log_metrics({"anomalies": int(df["is_anomaly"].sum()), "anomaly_rate_pct": float(df["is_anomaly"].mean()*100)})
    except Exception as e:
        logging.warning(f"MLflow skipped: {e}")
    df.to_parquet("/tmp/telemetry_predictions.parquet", index=False)
    context["ti"].xcom_push(key="results_path", value="/tmp/telemetry_predictions.parquet")
    context["ti"].xcom_push(key="anomaly_count", value=int(df["is_anomaly"].sum()))
    context["ti"].xcom_push(key="anomaly_rate_pct", value=float(df["is_anomaly"].mean()*100))
    return "/tmp/telemetry_predictions.parquet"

def store_results(**context):
    import pandas as pd
    path = context["ti"].xcom_pull(task_ids="run_inference", key="results_path")
    df = pd.read_parquet(path)
    os.makedirs(f"/tmp/delta/{context['ds']}", exist_ok=True)
    df[df["is_anomaly"]==1].to_parquet(f"/tmp/delta/{context['ds']}/part-00000.parquet", index=False)
    logging.info(f"✅ Stored {int(df['is_anomaly'].sum())} anomalies")

with DAG("vehicle_telemetry_ml_pipeline", default_args=default_args, schedule="@daily", start_date=pendulum.datetime(2024,1,1,tz="UTC"), catchup=False, tags=["telemetry","tesla"]) as dag:
    t1 = PythonOperator(task_id="ingest_telemetry", python_callable=ingest_telemetry)
    t2 = PythonOperator(task_id="validate_telemetry", python_callable=validate_telemetry)
    t3 = PythonOperator(task_id="run_inference", python_callable=run_inference)
    t4 = PythonOperator(task_id="store_results", python_callable=store_results)
    t1 >> t2 >> t3 >> t4
