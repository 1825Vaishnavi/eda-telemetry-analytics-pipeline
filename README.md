# 📡 EDA Tool Telemetry Analysis Pipeline

> An end-to-end data engineering and analytics pipeline for processing, analyzing, and visualizing performance telemetry from EDA tool executions — built on Azure Databricks, Python, SQL, and Power BI.

---

## 🎯 Problem Statement

EDA tools generate massive volumes of performance telemetry — runtime metrics, memory usage, job failure rates, and error patterns. Without automated analysis, engineers spend hours manually reviewing logs to detect regressions and anomalies.

This pipeline automates that process: from raw telemetry ingestion → anomaly detection → automated KPI dashboards → stakeholder-ready reports.

---

## 🏗️ Architecture

```
Raw Telemetry Logs
        ↓
[ Ingestion & Validation ] — Python + SQL
        ↓
[ Data Transformation ]  — Azure Databricks + PySpark
        ↓
[ Anomaly Detection ]    — Isolation Forest + Z-Score
        ↓
[ KPI Dashboard ]        — Power BI
        ↓
[ Automated Reports ]    — Scheduled weekly summaries
```

---

## ✨ Key Features

- **Automated ingestion pipeline** processing 1M+ telemetry records using Python and SQL
- **Anomaly detection** using Isolation Forest to flag performance regressions automatically
- **Trend analysis** — week-over-week runtime and failure rate tracking by tool type
- **Power BI dashboard** with 15+ KPI visualizations (runtime, memory, failure rate, error codes)
- **Automated weekly report** summarizing pipeline health for stakeholders
- **Dockerized pipeline** with GitHub Actions CI/CD for scheduled runs

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data Engineering | Python, SQL, Azure Databricks, PySpark |
| Anomaly Detection | Scikit-learn (Isolation Forest), NumPy |
| Visualization | Power BI, Matplotlib, Seaborn |
| Orchestration | GitHub Actions, Docker |
| Storage | PostgreSQL, Azure Blob Storage |

---

## 📊 Dashboard KPIs

- Average job runtime (by tool type, time period)
- Job failure rate & timeout trends
- Memory usage spikes
- Anomaly alert log (auto-flagged jobs)
- Week-over-week regression detection
- Error code distribution

---

## 📁 Project Structure

```
telemetry-analysis-pipeline/
│
├── data/
│   └── simulate_telemetry.py        # Generates synthetic telemetry logs
│
├── pipeline/
│   ├── ingest.py                    # Data ingestion & validation
│   ├── transform.py                 # Cleaning & feature engineering
│   ├── anomaly_detection.py         # Isolation Forest + Z-score detection
│   └── reporting.py                 # Automated weekly summary report
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard (local preview)
│
├── sql/
│   └── schema.sql                   # PostgreSQL schema definition
│
├── notebooks/
│   └── telemetry_analysis.ipynb     # Azure Databricks notebook
│
├── docker/
│   └── Dockerfile
│
├── .github/workflows/
│   └── pipeline.yml                 # GitHub Actions automation
│
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/1825Vaishnavi/telemetry-analysis-pipeline.git
cd telemetry-analysis-pipeline

# Install dependencies
pip install -r requirements.txt

# Generate synthetic telemetry data
python data/simulate_telemetry.py

# Run the pipeline
python pipeline/ingest.py
python pipeline/transform.py
python pipeline/anomaly_detection.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 📈 Results

- Automated detection of performance anomalies across 1M+ telemetry records
- Reduced manual report generation time by ~75%
- Isolation Forest detected regressions with high precision across tool types
- Power BI dashboard enabled real-time stakeholder visibility into pipeline health

---

## 🔗 Related Skills Demonstrated

`Azure Databricks` `PySpark` `Python` `SQL` `Power BI` `Anomaly Detection` `ETL Pipelines` `Docker` `CI/CD` `Data Engineering`

---

## 📫 Author

**Vaishnavi Gajarla**  
MS Data Analytics Engineering @ Northeastern University  
[LinkedIn](https://www.linkedin.com/in/vaishnavi-mallikarjun-gajarla-726323296) | [GitHub](https://github.com/1825Vaishnavi) | gajarla.v@northeastern.edu
