"""
lstm_model.py
PyTorch LSTM — Sequential Anomaly Detection on Vehicle Telemetry
Trains 5 configs, logs to MLflow, registers best model.

Usage:
    python src/lstm_model.py
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SENSOR_COLS = [
    "speed_kmh", "battery_pct", "temp_motor_c", "temp_battery_c",
    "regen_brake_kw", "accel_x", "accel_y", "accel_z",
    "voltage_v", "current_a"
]
SEQ_LEN     = 30        # 30-timestep sliding window
N_FEATURES  = len(SENSOR_COLS)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR   = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 5 experiment configs for MLflow
EXPERIMENT_CONFIGS = [
    {"hidden_size": 32,  "num_layers": 1, "dropout": 0.1, "lr": 1e-3, "batch_size": 64,  "epochs": 5},
    {"hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64,  "epochs": 5},
    {"hidden_size": 128, "num_layers": 2, "dropout": 0.3, "lr": 5e-4, "batch_size": 128, "epochs": 5},
    {"hidden_size": 64,  "num_layers": 3, "dropout": 0.2, "lr": 1e-3, "batch_size": 32,  "epochs": 5},
    {"hidden_size": 128, "num_layers": 3, "dropout": 0.4, "lr": 1e-4, "batch_size": 128, "epochs": 5},
]


# ─────────────────────────────────────────────
# DATA GENERATION
# Simulates real telemetry with injected anomalies
# In production: replace with Delta Lake / Parquet reads
# ─────────────────────────────────────────────
def generate_telemetry(n_records=50_000):
    log.info(f"Generating {n_records:,} telemetry records...")
    np.random.seed(42)

    data = {
        "speed_kmh":      np.random.normal(80, 30, n_records).clip(0, 200),
        "battery_pct":    np.random.normal(65, 20, n_records).clip(0, 100),
        "temp_motor_c":   np.random.normal(75, 15, n_records).clip(20, 150),
        "temp_battery_c": np.random.normal(30, 10, n_records).clip(-20, 80),
        "regen_brake_kw": np.random.exponential(5, n_records).clip(0, 50),
        "accel_x":        np.random.normal(0, 2, n_records),
        "accel_y":        np.random.normal(0, 2, n_records),
        "accel_z":        np.random.normal(9.8, 0.5, n_records),
        "voltage_v":      np.random.normal(400, 20, n_records).clip(300, 500),
        "current_a":      np.random.normal(100, 40, n_records).clip(-200, 500),
    }
    df = pd.DataFrame(data)

    # Inject anomalies (~5% of records)
    labels = np.zeros(n_records, dtype=int)
    anomaly_idx = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
    df.loc[anomaly_idx, "temp_motor_c"]  += np.random.uniform(40, 80, len(anomaly_idx))
    df.loc[anomaly_idx, "battery_pct"]   -= np.random.uniform(30, 60, len(anomaly_idx))
    df.loc[anomaly_idx, "voltage_v"]     += np.random.uniform(50, 100, len(anomaly_idx))
    labels[anomaly_idx] = 1

    df["label"] = labels
    log.info(f"Anomaly rate: {labels.mean()*100:.1f}%")
    return df


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class TelemetryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]          # (seq_len, n_features)
        y_label = self.y[idx + self.seq_len]              # label at end of window
        return x_seq, y_label


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class TelemetryLSTM(nn.Module):
    """
    Bidirectional LSTM for sequential anomaly detection.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 2)  — normal / anomaly logits
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm     = nn.LayerNorm(hidden_size * 2)
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)           # (batch, seq_len, hidden*2)
        out     = out[:, -1, :]         # last timestep
        out     = self.norm(out)
        out     = self.dropout(out)
        return self.fc(out)             # (batch, 2)


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits     = model(X_batch)
            loss       = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n          = len(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss":      total_loss / n,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "preds":     all_preds,
        "labels":    all_labels,
    }


# ─────────────────────────────────────────────
# TRAIN ONE CONFIG
# ─────────────────────────────────────────────
def train_config(config, train_loader, val_loader, run_name):
    model = TelemetryLSTM(
        input_size=N_FEATURES,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(DEVICE)

    # Class weights to handle imbalance (95% normal, 5% anomaly)
    class_weights = torch.tensor([1.0, 10.0]).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    best_f1, best_state = 0.0, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_metrics           = evaluate(model, val_loader, criterion)
        scheduler.step(val_metrics["loss"])

        log.info(
            f"[{run_name}] Epoch {epoch+1}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.3f}"
        )

        mlflow.log_metrics({
            "train_loss":  train_loss,
            "train_acc":   train_acc,
            "val_loss":    val_metrics["loss"],
            "val_accuracy":val_metrics["accuracy"],
            "val_f1":      val_metrics["f1"],
        }, step=epoch)

        if val_metrics["f1"] > best_f1:
            best_f1   = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return model, best_f1, evaluate(model, val_loader, criterion)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info(f"Using device: {DEVICE}")

    # ── Data ───────────────────────────────────────────────────────────
    df = generate_telemetry(n_records=50_000)
    X  = df[SENSOR_COLS].values
    y  = df["label"].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / val split (80/20)
    split    = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y[:split], y[split:]

    # ── MLflow setup ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment("vehicle-telemetry-lstm")

    best_run_id, best_f1_overall, best_model, best_config = None, 0.0, None, None

    # ── Run 5 configs ──────────────────────────────────────────────────
    for i, config in enumerate(EXPERIMENT_CONFIGS):
        run_name = f"lstm_config_{i+1}_h{config['hidden_size']}_l{config['num_layers']}"
        log.info(f"\n{'='*60}")
        log.info(f"Running config {i+1}/5: {config}")
        log.info(f"{'='*60}")

        train_ds = TelemetryDataset(X_train, y_train)
        val_ds   = TelemetryDataset(X_val,   y_val)

        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"],
            shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=config["batch_size"] * 2,
            shuffle=False, num_workers=0
        )

        with mlflow.start_run(run_name=run_name) as run:
            # Log all hyperparameters
            mlflow.log_params({
                "model_type":    "BiLSTM",
                "hidden_size":   config["hidden_size"],
                "num_layers":    config["num_layers"],
                "dropout":       config["dropout"],
                "learning_rate": config["lr"],
                "batch_size":    config["batch_size"],
                "epochs":        config["epochs"],
                "seq_len":       SEQ_LEN,
                "n_features":    N_FEATURES,
                "optimizer":     "Adam",
                "scheduler":     "ReduceLROnPlateau",
                "class_weight_anomaly": 10.0,
            })

            model, best_f1, final_metrics = train_config(
                config, train_loader, val_loader, run_name
            )

            # Log final metrics
            mlflow.log_metrics({
                "final_accuracy":  final_metrics["accuracy"],
                "final_precision": final_metrics["precision"],
                "final_recall":    final_metrics["recall"],
                "final_f1":        final_metrics["f1"],
                "best_val_f1":     best_f1,
            })

            log.info(f"\n📊 Config {i+1} Results:")
            log.info(f"   Accuracy:  {final_metrics['accuracy']*100:.1f}%")
            log.info(f"   Precision: {final_metrics['precision']*100:.1f}%")
            log.info(f"   Recall:    {final_metrics['recall']*100:.1f}%")
            log.info(f"   F1 Score:  {final_metrics['f1']*100:.1f}%")

            # Save model artifact
            model_path = os.path.join(MODEL_DIR, f"lstm_config_{i+1}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config":           config,
                "scaler_mean":      scaler.mean_.tolist(),
                "scaler_std":       scaler.scale_.tolist(),
                "sensor_cols":      SENSOR_COLS,
                "seq_len":          SEQ_LEN,
                "final_f1":         final_metrics["f1"],
            }, model_path)
            mlflow.log_artifact(model_path)

            # Log PyTorch model to MLflow
            # mlflow.pytorch.log_model skipped - version compatibility

            if best_f1 > best_f1_overall:
                best_f1_overall = best_f1
                best_run_id = run.info.run_id; best_config = config
                best_model = TelemetryLSTM(N_FEATURES, config["hidden_size"], config["num_layers"], config["dropout"]).to(DEVICE); best_model.load_state_dict({k: v.cpu().clone() for k, v in model.state_dict().items()})

    # ── Register best model ────────────────────────────────────────────
    log.info(f"\n🏆 Best model: run_id={best_run_id}, F1={best_f1_overall:.4f}")

    model_uri = f"runs:/{best_run_id}/lstm_model"
    try:
        registered = mlflow.register_model(
            model_uri=model_uri,
            name="telemetry-lstm-anomaly-detector"
        )
        log.info(f"✅ Model registered: version {registered.version}")

        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="telemetry-lstm-anomaly-detector",
            version=registered.version,
            stage="Production",
        )
        log.info("✅ Model promoted to Production stage")
    except Exception as e:
        log.warning(f"⚠️ Model registry skipped: {e}")

    # ── Save best model locally ────────────────────────────────────────
    best_path = os.path.join(MODEL_DIR, "lstm_best.pt")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "best_f1":          best_f1_overall,
        "sensor_cols":      SENSOR_COLS,
        "seq_len":          SEQ_LEN,
        "scaler_mean":      scaler.mean_.tolist(),
        "scaler_std":       scaler.scale_.tolist(),
    }, best_path)
    log.info(f"✅ Best model saved → {best_path}")

    # ── Final report ───────────────────────────────────────────────────
    final_metrics = evaluate(best_model, val_loader, nn.CrossEntropyLoss())
    log.info("\n" + "="*60)
    log.info("FINAL BEST MODEL REPORT")
    log.info("="*60)
    log.info(f"Accuracy:  {final_metrics['accuracy']*100:.2f}%")
    log.info(f"Precision: {final_metrics['precision']*100:.2f}%")
    log.info(f"Recall:    {final_metrics['recall']*100:.2f}%")
    log.info(f"F1 Score:  {final_metrics['f1']*100:.2f}%")
    log.info("\nClassification Report:")
    log.info(classification_report(
        final_metrics["labels"], final_metrics["preds"],
        target_names=["Normal", "Anomaly"]
    ))
    log.info("="*60)
    log.info("✅ LSTM training complete! Check MLflow at http://localhost:5000")


if __name__ == "__main__":
    main()
