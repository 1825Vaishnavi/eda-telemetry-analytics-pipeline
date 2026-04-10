"""
spark_ingestion.py
PySpark + Delta Lake — Vehicle Telemetry Ingestion & Transformation
Mirrors Tesla/Databricks production architecture.

Runs in 2 modes:
  1. LOCAL  — uses local PySpark (for development/portfolio demo)
  2. DATABRICKS — same code, just change SPARK_MODE env var

Usage (local):
    python src/spark_ingestion.py

Usage (Databricks notebook):
    - Upload this file to Databricks workspace
    - Change SPARK_MODE = "databricks"
    - Attach to a cluster and run
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_MODE     = os.getenv("SPARK_MODE", "local")          # "local" or "databricks"
DELTA_PATH     = os.getenv("DELTA_PATH", "./data/delta")   # local Delta Lake path
RAW_PATH       = os.getenv("RAW_PATH",   "./data/raw")
PROCESSED_PATH = os.getenv("PROCESSED_PATH", "./data/processed")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "telemetry-data")
AZURE_ACCOUNT   = os.getenv("AZURE_ACCOUNT",   "yourstorageaccount")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN",  "")        # set in production

os.makedirs(DELTA_PATH,     exist_ok=True)
os.makedirs(RAW_PATH,       exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)


# ─────────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────────
def get_spark():
    """
    Returns a SparkSession.
    - Local mode:      starts embedded PySpark
    - Databricks mode: uses existing 'spark' variable from cluster
    """
    if SPARK_MODE == "databricks":
        # On Databricks, 'spark' is pre-created — just return it
        # This line runs in Databricks notebook environment
        return spark  # noqa: F821

    # Local mode
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("VehicleTelemetryPipeline")
        .master("local[*]")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")   # small for local dev
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info("✅ Local SparkSession started")
    return spark


# ─────────────────────────────────────────────
# AZURE BLOB CONFIG (production)
# ─────────────────────────────────────────────
def configure_azure_storage(spark):
    """
    Configure Spark to read from Azure Blob Storage.
    In production: set AZURE_SAS_TOKEN environment variable.
    In local dev:  this is a no-op.
    """
    if SPARK_MODE == "databricks" and AZURE_SAS_TOKEN:
        spark.conf.set(
            f"fs.azure.sas.{AZURE_CONTAINER}.{AZURE_ACCOUNT}.blob.core.windows.net",
            AZURE_SAS_TOKEN
        )
        log.info("✅ Azure Blob Storage configured")
    else:
        log.info("ℹ️  Azure storage not configured — using local paths")


# ─────────────────────────────────────────────
# STEP 1: GENERATE / INGEST RAW DATA
# ─────────────────────────────────────────────
def ingest_raw_data(spark, n_records=1_000_000):
    """
    Ingests raw telemetry data.
    Production: reads from Azure Blob Storage (abfss:// path)
    Local:      generates simulated data and writes to parquet
    """
    log.info(f"🚗 [INGEST] Generating {n_records:,} telemetry records...")

    # ── Production path (Azure Blob) ──────────────────────────────────
    if SPARK_MODE == "databricks" and AZURE_SAS_TOKEN:
        azure_path = (
            f"wasbs://{AZURE_CONTAINER}@{AZURE_ACCOUNT}"
            f".blob.core.windows.net/raw/telemetry/"
        )
        df = spark.read.parquet(azure_path)
        log.info(f"✅ Read {df.count():,} records from Azure Blob")
        return df

    # ── Local simulation ───────────────────────────────────────────────
    np.random.seed(42)
    pdf = pd.DataFrame({
        "vehicle_id":     np.random.choice(
                              [f"VH_{i:05d}" for i in range(10000)], n_records),
        "event_ts":       pd.date_range("2024-01-01", periods=n_records, freq="1s")
                              .astype(str),
        "speed_kmh":      np.random.normal(80,  30,  n_records).clip(0,   200),
        "battery_pct":    np.random.normal(65,  20,  n_records).clip(0,   100),
        "temp_motor_c":   np.random.normal(75,  15,  n_records).clip(20,  150),
        "temp_battery_c": np.random.normal(30,  10,  n_records).clip(-20, 80),
        "regen_brake_kw": np.random.exponential(5,   n_records).clip(0,   50),
        "accel_x":        np.random.normal(0,   2,   n_records),
        "accel_y":        np.random.normal(0,   2,   n_records),
        "accel_z":        np.random.normal(9.8, 0.5, n_records),
        "voltage_v":      np.random.normal(400, 20,  n_records).clip(300, 500),
        "current_a":      np.random.normal(100, 40,  n_records).clip(-200,500),
        "odometer_km":    np.random.uniform(0,  200000, n_records),
        "region":         np.random.choice(
                              ["US-CA", "US-TX", "US-NY", "EU-DE", "EU-FR"], n_records),
        "firmware_ver":   np.random.choice(
                              ["v12.1", "v12.2", "v12.3", "v13.0"], n_records),
    })

    # Inject anomalies
    anomaly_idx = np.random.choice(n_records, size=int(n_records * 0.03), replace=False)
    pdf.loc[anomaly_idx, "temp_motor_c"]  += np.random.uniform(40, 80,  len(anomaly_idx))
    pdf.loc[anomaly_idx, "battery_pct"]   -= np.random.uniform(30, 60,  len(anomaly_idx))
    pdf.loc[anomaly_idx, "voltage_v"]     += np.random.uniform(50, 100, len(anomaly_idx))
    pdf["injected_anomaly"] = 0
    pdf.loc[anomaly_idx, "injected_anomaly"] = 1

    # Save raw parquet
    raw_file = os.path.join(RAW_PATH, "telemetry_raw.parquet")
    pdf.to_parquet(raw_file, index=False)

    # Load into Spark
    df = spark.read.parquet(raw_file)
    log.info(f"✅ [INGEST] {df.count():,} records loaded into Spark")
    return df


# ─────────────────────────────────────────────
# STEP 2: PYSPARK TRANSFORMATIONS
# ─────────────────────────────────────────────
def transform_data(spark, df):
    """
    PySpark transformations — mirrors Databricks production pipeline:
    - Cast types
    - Derive features (power_kw, temp_delta, accel_magnitude)
    - Add anomaly flags
    - Partition by date + region
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType,
        DoubleType, IntegerType, TimestampType
    )
    from pyspark.sql.window import Window

    log.info("⚡ [TRANSFORM] Running PySpark transformations...")

    # ── Cast + clean ───────────────────────────────────────────────────
    df = (df
        .withColumn("event_ts",       F.to_timestamp("event_ts"))
        .withColumn("speed_kmh",      F.col("speed_kmh").cast(DoubleType()))
        .withColumn("battery_pct",    F.col("battery_pct").cast(DoubleType()))
        .withColumn("temp_motor_c",   F.col("temp_motor_c").cast(DoubleType()))
        .withColumn("temp_battery_c", F.col("temp_battery_c").cast(DoubleType()))
        .withColumn("voltage_v",      F.col("voltage_v").cast(DoubleType()))
        .withColumn("current_a",      F.col("current_a").cast(DoubleType()))
    )

    # ── Derived features ───────────────────────────────────────────────
    df = (df
        # Instantaneous power (kW)
        .withColumn("power_kw",
            F.round(F.col("voltage_v") * F.col("current_a") / 1000.0, 2))

        # Temperature delta (motor vs battery)
        .withColumn("temp_delta_c",
            F.round(F.col("temp_motor_c") - F.col("temp_battery_c"), 2))

        # 3D acceleration magnitude
        .withColumn("accel_magnitude",
            F.round(F.sqrt(
                F.col("accel_x")**2 +
                F.col("accel_y")**2 +
                F.col("accel_z")**2
            ), 4))

        # Energy efficiency (speed per kW)
        .withColumn("efficiency_km_per_kw",
            F.when(F.col("power_kw") > 0,
                F.round(F.col("speed_kmh") / F.col("power_kw"), 4)
            ).otherwise(F.lit(None)))

        # Date partition columns
        .withColumn("date_",   F.to_date("event_ts"))
        .withColumn("year_",   F.year("event_ts"))
        .withColumn("month_",  F.month("event_ts"))
        .withColumn("hour_",   F.hour("event_ts"))
    )

    # ── Statistical anomaly flags (rule-based, Layer 0) ───────────────
    df = (df
        .withColumn("flag_high_temp",
            F.when(F.col("temp_motor_c") > 130, 1).otherwise(0))
        .withColumn("flag_low_battery",
            F.when(F.col("battery_pct") < 10, 1).otherwise(0))
        .withColumn("flag_overvoltage",
            F.when(F.col("voltage_v") > 480, 1).otherwise(0))
        .withColumn("flag_high_accel",
            F.when(F.col("accel_magnitude") > 15, 1).otherwise(0))
        .withColumn("rule_anomaly",
            F.when(
                (F.col("flag_high_temp") == 1) |
                (F.col("flag_low_battery") == 1) |
                (F.col("flag_overvoltage") == 1) |
                (F.col("flag_high_accel") == 1),
                1
            ).otherwise(0))
    )

    # ── Per-vehicle rolling stats (window function) ────────────────────
    w_vehicle = (Window
        .partitionBy("vehicle_id")
        .orderBy("event_ts")
        .rowsBetween(-29, 0))   # 30-row rolling window

    df = (df
        .withColumn("rolling_avg_temp",
            F.round(F.avg("temp_motor_c").over(w_vehicle), 2))
        .withColumn("rolling_avg_battery",
            F.round(F.avg("battery_pct").over(w_vehicle), 2))
        .withColumn("rolling_std_speed",
            F.round(F.stddev("speed_kmh").over(w_vehicle), 4))
    )

    log.info(f"✅ [TRANSFORM] Features engineered. Schema:")
    df.printSchema()

    return df


# ─────────────────────────────────────────────
# STEP 3: WRITE TO DELTA LAKE
# ─────────────────────────────────────────────
def write_delta(df, path, partition_cols=None, mode="overwrite"):
    """
    Write DataFrame to Delta Lake.
    Production: path = abfss://container@account.dfs.core.windows.net/delta/telemetry
    Local:      path = ./data/delta/telemetry
    """
    try:
        writer = df.write.format("delta").mode(mode)
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.save(path)
        log.info(f"✅ [DELTA] Written to {path} (mode={mode})")
    except Exception as e:
        # Delta Lake not installed — fallback to parquet
        log.warning(f"⚠️  Delta write failed ({e}), falling back to parquet")
        writer = df.write.mode(mode)
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.parquet(path + "_parquet")
        log.info(f"✅ [PARQUET] Written to {path}_parquet")


# ─────────────────────────────────────────────
# STEP 4: AGGREGATIONS FOR DASHBOARD
# ─────────────────────────────────────────────
def compute_aggregations(spark, df):
    """
    Compute fleet-level KPIs for Power BI / Streamlit dashboard.
    """
    from pyspark.sql import functions as F

    log.info("📊 [AGGREGATIONS] Computing fleet KPIs...")

    # Fleet summary per vehicle per hour
    fleet_agg = (df
        .groupBy("vehicle_id", "year_", "month_", "hour_", "region")
        .agg(
            F.avg("speed_kmh")       .alias("avg_speed_kmh"),
            F.avg("battery_pct")     .alias("avg_battery_pct"),
            F.avg("temp_motor_c")    .alias("avg_temp_motor_c"),
            F.avg("power_kw")        .alias("avg_power_kw"),
            F.sum("rule_anomaly")    .alias("anomaly_count"),
            F.count("*")             .alias("record_count"),
            F.max("odometer_km")     .alias("max_odometer_km"),
        )
    )

    # Anomaly summary by region
    region_agg = (df
        .groupBy("region", "date_")
        .agg(
            F.count("*")          .alias("total_records"),
            F.sum("rule_anomaly") .alias("total_anomalies"),
            F.avg("battery_pct")  .alias("avg_battery_pct"),
            F.avg("temp_motor_c") .alias("avg_temp_motor_c"),
        )
        .withColumn("anomaly_rate",
            F.round(F.col("total_anomalies") / F.col("total_records"), 4))
    )

    # Save aggregations
    fleet_path  = os.path.join(PROCESSED_PATH, "fleet_agg")
    region_path = os.path.join(PROCESSED_PATH, "region_agg")

    fleet_agg.write.mode("overwrite").parquet(fleet_path)
    region_agg.write.mode("overwrite").parquet(region_path)

    log.info(f"✅ Fleet aggregation:  {fleet_agg.count():,} rows → {fleet_path}")
    log.info(f"✅ Region aggregation: {region_agg.count():,} rows → {region_path}")

    return fleet_agg, region_agg


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    log.info("="*60)
    log.info("VEHICLE TELEMETRY SPARK PIPELINE")
    log.info(f"Mode: {SPARK_MODE.upper()}")
    log.info("="*60)

    spark = get_spark()
    configure_azure_storage(spark)

    # Step 1: Ingest
    raw_df = ingest_raw_data(spark, n_records=100_000)  # use 100k locally

    # Step 2: Transform
    transformed_df = transform_data(spark, raw_df)

    # Step 3: Write to Delta Lake
    delta_telemetry_path = os.path.join(DELTA_PATH, "telemetry")
    write_delta(
        transformed_df,
        path=delta_telemetry_path,
        partition_cols=["year_", "month_", "region"],
        mode="overwrite"
    )

    # Step 4: Aggregations
    fleet_agg, region_agg = compute_aggregations(spark, transformed_df)

    # Print sample
    log.info("\n📋 Sample transformed records:")
    transformed_df.select(
        "vehicle_id", "speed_kmh", "battery_pct", "temp_motor_c",
        "power_kw", "temp_delta_c", "accel_magnitude", "rule_anomaly"
    ).show(5, truncate=False)

    log.info("\n📋 Region anomaly summary:")
    region_agg.orderBy("anomaly_rate", ascending=False).show(10)

    # Summary stats
    total     = transformed_df.count()
    anomalies = transformed_df.filter("rule_anomaly = 1").count()
    log.info("\n" + "="*60)
    log.info("PIPELINE SUMMARY")
    log.info("="*60)
    log.info(f"Total records:    {total:,}")
    log.info(f"Anomalies found:  {anomalies:,} ({anomalies/total*100:.2f}%)")
    log.info(f"Delta path:       {delta_telemetry_path}")
    log.info(f"Processed path:   {PROCESSED_PATH}")
    log.info("="*60)
    log.info("✅ Spark pipeline complete!")

    spark.stop()


if __name__ == "__main__":
    main()
