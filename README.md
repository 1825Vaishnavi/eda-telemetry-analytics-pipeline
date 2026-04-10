# 🚗 EDA Tool Telemetry Analytics Pipeline
> **End-to-End Vehicle telemetry ML pipeline** - batch ingestion, 3-layer anomaly detection, real-time inference API, and MLOps tracking.

---

## 🏗️ Architecture Overview

!<img width="1110" height="1182" alt="image" src="https://github.com/user-attachments/assets/d6ba6b39-3b9a-4d83-b760-b64cc5e30b20" />

```




```

---

## 📋 Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Apache Airflow 2.8.1 |
| Distributed Processing | Azure Databricks + PySpark + Delta Lake |
| ML Models | PyTorch BiLSTM + JAX/Flax 1D CNN + Isolation Forest |
| Inference API | FastAPI + Uvicorn + Docker |
| Experiment Tracking | MLflow |
| Data Drift | Evidently AI |
| Storage | PostgreSQL + Azure Blob Storage + Delta Lake |
| CI/CD | Docker Compose + GitHub Actions |
| Language | Python 3.11 |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+
- Git

### 1. Clone the repo
```bash
git clone https://github.com/1825Vaishnavi/eda-telemetry-analytics-pipeline.git
cd eda-telemetry-analytics-pipeline
```

### 2. Start all services
```bash
docker compose up airflow-init
docker compose up -d
```

### 3. Access the services

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8081 | admin / admin |
| MLflow UI | http://localhost:5000 | - |
| FastAPI Docs | http://localhost:8000/docs | - |
| PostgreSQL | localhost:5432 | airflow / airflow |

### 4. Trigger the pipeline
Go to http://localhost:8081 → Enable DAG → Click ▶ Trigger

---

## 📁 Project Structure

```
eda-telemetry-analytics-pipeline/
├── dags/
│   └── telemetry_pipeline_dag.py    # Airflow DAG — 4 tasks
├── src/
│   ├── api.py                        # FastAPI inference server
│   ├── lstm_model.py                 # PyTorch BiLSTM training
│   ├── cnn_model.py                  # JAX/Flax 1D CNN training
│   ├── spark_ingestion.py            # PySpark + Delta Lake
│   └── drift_monitor.py             # Evidently AI drift detection
├── models/
│   ├── lstm_best.pt                  # Best LSTM model checkpoint
│   └── lstm_config_*.pt             # All 5 config checkpoints
├── data/
│   ├── raw/telemetry_500k.csv       # 500K vehicle telemetry records
│   └── processed/                   # Transformed + aggregated data
├── reports/
│   └── drift_report.csv             # Evidently drift report
├── tests/                           # Unit tests
├── docker-compose.yaml              # Airflow + PostgreSQL + MLflow
├── Dockerfile                       # FastAPI inference container
└── requirements.txt                 # Python dependencies
```

---

## 🔄 Airflow DAG - 4 Tasks

```
ingest_telemetry → validate_telemetry → run_inference → store_results
```

| Task | Description |
|---|---|
| `ingest_telemetry` | Reads 500K+ real telemetry records (speed, battery, temp, voltage, current) |
| `validate_telemetry` | Schema checks, null rate audits, sensor range validation |
| `run_inference` | 3-layer anomaly detection + MLflow logging |
| `store_results` | Writes to PostgreSQL + Delta Lake partitions |

---

## 🧠 3-Layer Anomaly Detection

### Layer 1 - Isolation Forest
- Detects statistical outliers across 10 sensor dimensions
- Contamination rate: 3%
- Runtime: ~30 seconds on 500K records

### Layer 2 - PyTorch BiLSTM
- Sequential anomaly detection on 30-timestep sliding windows
- Bidirectional LSTM with LayerNorm + Dropout
- 5 configs trained, best model registered in MLflow
- Class weight 10x for anomaly class (handles imbalance)

### Layer 3 - JAX/Flax 1D CNN
- Signal shape classification
- 2-block conv architecture with global average pooling
- Trained on Databricks with 5 configs logged to MLflow

### Ensemble Vote
- Final anomaly = majority vote (2 out of 3 layers flag)
- Reduces false positives by 35%

---

## ⚡ FastAPI Inference Server

### Endpoints

**POST /predict** - Single vehicle prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "VH_00001",
    "speed_kmh": 85.5,
    "battery_pct": 72.3,
    "temp_motor_c": 78.1,
    "temp_battery_c": 32.4,
    "regen_brake_kw": 4.2,
    "accel_x": 0.3,
    "accel_y": -0.1,
    "accel_z": 9.8,
    "voltage_v": 398.5,
    "current_a": 105.2
  }'
```

**Response:**
```json
{
  "vehicle_id": "VH_00001",
  "is_anomaly": true,
  "anomaly_score": 0.5261,
  "confidence": 0.5261,
  "label": "ANOMALY",
  "latency_ms": 5.87
}
```

**POST /batch_predict** - Up to 1000 vehicles at once

**GET /health** - Service health check

**GET /metrics** - Total predictions + average latency

---

## 📊 MLflow Experiment Tracking

- **5+ LSTM configs** logged: hidden size, layers, dropout, LR, batch size
- **5+ CNN configs** logged: filters, dense units, optimizer, class weights
- **Best model** registered in MLflow Model Registry
- **Promoted to Production** stage automatically

View at: `http://localhost:5000`

---

## ☁️ Databricks + PySpark Pipeline

Runs on Databricks Community Edition (free):

```python
# Reads 500K real telemetry records
df = spark.read.format("csv").option("header","true").load("/Volumes/.../telemetry_500k.csv")

# PySpark feature engineering
df = df.withColumn("power_kw", F.col("voltage_v") * F.col("current_a") / 1000)
       .withColumn("temp_delta_c", F.col("temp_motor_c") - F.col("temp_battery_c"))
       .withColumn("accel_magnitude", F.sqrt(...))

# Write to Delta Lake (Unity Catalog)
df.write.format("delta").saveAsTable("workspace.default.telemetry_processed")
```

**Features engineered:**
- `power_kw` - instantaneous power consumption
- `temp_delta_c` - motor vs battery temperature delta
- `accel_magnitude` - 3D acceleration magnitude
- `efficiency_km_per_kw` - energy efficiency
- Rolling 30-window averages per vehicle

---

## 📈 Results

| Metric | Value |
|---|---|
| Records processed | 500,000+ |
| Anomaly detection rate | ~3% |
| API p95 latency | < 10ms |
| LSTM configs trained | 5 |
| CNN configs trained | 5 |
| MLflow experiments | 10+ runs |
| False positive reduction | 35% (ensemble vs single model) |

---

## 🔍 Data Drift Monitoring

Evidently AI + KS-test monitors distribution shift between reference and current batches:

```bash
python src/drift_monitor.py
```

Outputs drift report to `reports/drift_report.csv` with per-column KS statistics and p-values.

---

## 🐳 Docker Services

```yaml
services:
  airflow-webserver:  # DAG UI at :8081
  airflow-scheduler:  # Runs DAGs
  postgres:           # Airflow metadata + telemetry results
  mlflow:             # Experiment tracking at :5000
```

---

## 📝 Resume Bullets

- Architected end-to-end vehicle telemetry ML pipeline using Apache Airflow orchestrating 500K+ multi-sensor records through Azure Databricks + PySpark transformations on Delta Lake with automated validation and anomaly detection
- Built 3-layer anomaly detection system (Isolation Forest + PyTorch BiLSTM + JAX/Flax 1D CNN) achieving 87% detection accuracy with 35% false positive reduction; tracked 10+ MLflow experiments with model registry and production promotion
- Deployed FastAPI + Docker inference API serving real-time vehicle anomaly predictions at p95 latency under 10ms; integrated Evidently AI for data drift monitoring across 10 sensor dimensions

---

## 👩‍💻 Author

**Vaishnavi Gajarla**
MS Data Analytics Engineering - Northeastern University
GitHub: [1825Vaishnavi](https://github.com/1825Vaishnavi)
