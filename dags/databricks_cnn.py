# ============================================================
# JAX/FLAX 1D CNN — Signal Shape Classification
# Databricks Notebook — Paste ALL in ONE cell, Run All
# ============================================================

import os
import numpy as np
import pandas as pd
import mlflow
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

print(f"✅ JAX version: {jax.__version__}")
print(f"✅ JAX backend: {jax.default_backend()}")

# MLflow setup — Databricks path
current_user = "vaishnavigajrala@gmail.com"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"/Users/{current_user}/vehicle-telemetry-cnn-jax")
print(f"✅ MLflow experiment set for {current_user}")

SENSOR_COLS = [
    "speed_kmh", "battery_pct", "temp_motor_c", "temp_battery_c",
    "regen_brake_kw", "accel_x", "accel_y", "accel_z",
    "voltage_v", "current_a"
]
SEQ_LEN    = 30
N_FEATURES = len(SENSOR_COLS)

# 5 CNN configs for MLflow
CNN_CONFIGS = [
    {"filters1": 16,  "filters2": 32,  "dense": 32,  "lr": 1e-3, "batch_size": 64,  "epochs": 3},
    {"filters1": 32,  "filters2": 64,  "dense": 64,  "lr": 1e-3, "batch_size": 64,  "epochs": 3},
    {"filters1": 64,  "filters2": 128, "dense": 64,  "lr": 5e-4, "batch_size": 128, "epochs": 3},
    {"filters1": 32,  "filters2": 64,  "dense": 128, "lr": 1e-3, "batch_size": 32,  "epochs": 3},
    {"filters1": 64,  "filters2": 128, "dense": 128, "lr": 1e-4, "batch_size": 128, "epochs": 3},
]

# ============================================================
# STEP 1: LOAD DATA FROM DELTA TABLE
# ============================================================
print("🚗 Loading telemetry data from Delta table...")

try:
    df = spark.sql("SELECT * FROM workspace.default.telemetry_processed LIMIT 30000")
    pdf = df.select(SENSOR_COLS + ["rule_anomaly"]).toPandas()
    pdf = pdf.rename(columns={"rule_anomaly": "label"})
    print(f"✅ Loaded {len(pdf):,} records from Delta table")
except Exception as e:
    print(f"⚠️ Delta read failed: {e}, generating data...")
    np.random.seed(42)
    n = 30000
    pdf = pd.DataFrame({
        "speed_kmh":      np.random.normal(80, 30, n).clip(0, 200),
        "battery_pct":    np.random.normal(65, 20, n).clip(0, 100),
        "temp_motor_c":   np.random.normal(75, 15, n).clip(20, 150),
        "temp_battery_c": np.random.normal(30, 10, n).clip(-20, 80),
        "regen_brake_kw": np.random.exponential(5, n).clip(0, 50),
        "accel_x":        np.random.normal(0, 2, n),
        "accel_y":        np.random.normal(0, 2, n),
        "accel_z":        np.random.normal(9.8, 0.5, n),
        "voltage_v":      np.random.normal(400, 20, n).clip(300, 500),
        "current_a":      np.random.normal(100, 40, n).clip(-200, 500),
        "label":          np.zeros(n, dtype=int),
    })
    idx = np.random.choice(n, size=int(n*0.05), replace=False)
    pdf.loc[idx, "temp_motor_c"] += 50
    pdf.loc[idx, "label"] = 1

# Prepare data
X = pdf[SENSOR_COLS].fillna(0).values
y = pdf["label"].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

# Make sequences
def make_sequences(X, y, seq_len=SEQ_LEN):
    seqs, lbls = [], []
    for i in range(len(X) - seq_len):
        seqs.append(X[i:i+seq_len])
        lbls.append(y[i+seq_len])
    return np.array(seqs, dtype=np.float32), np.array(lbls, dtype=np.int32)

X_seq, y_seq = make_sequences(X_scaled, y)
split = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

print(f"✅ Train: {len(X_train):,} | Val: {len(X_val):,}")
print(f"✅ Anomaly rate: {y_seq.mean()*100:.1f}%")


# ============================================================
# STEP 2: JAX/FLAX CNN MODEL
# ============================================================
class TelemetryCNN(nn.Module):
    filters1: int
    filters2: int
    dense:    int

    @nn.compact
    def __call__(self, x, training: bool = False):
        # Block 1
        x = nn.Conv(features=self.filters1, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.filters1, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        # Block 2
        x = nn.Conv(features=self.filters2, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.filters2, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = x.mean(axis=1)  # global average pool
        # Dense
        x = nn.Dense(self.dense)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.3)(x, deterministic=not training)
        x = nn.Dense(2)(x)
        return x

def cross_entropy_loss(logits, labels):
    weights  = jnp.where(labels == 1, 10.0, 1.0)
    one_hot  = jax.nn.one_hot(labels, num_classes=2)
    log_soft = jax.nn.log_softmax(logits)
    loss     = -jnp.sum(one_hot * log_soft, axis=-1)
    return jnp.mean(loss * weights)

@jax.jit
def train_step(state, batch_x, batch_y, dropout_rng):
    def loss_fn(params):
        logits = state.apply_fn(params, batch_x, training=True,
                                rngs={"dropout": dropout_rng})
        return cross_entropy_loss(logits, batch_y), logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    preds = jnp.argmax(logits, axis=-1)
    acc   = jnp.mean(preds == batch_y)
    return state, loss, acc

@jax.jit
def eval_step(state, batch_x, batch_y):
    logits = state.apply_fn(state.params, batch_x, training=False)
    loss   = cross_entropy_loss(logits, batch_y)
    preds  = jnp.argmax(logits, axis=-1)
    acc    = jnp.mean(preds == batch_y)
    return loss, acc, preds


# ============================================================
# STEP 3: TRAIN 5 CONFIGS
# ============================================================
best_f1_overall  = 0.0
best_config_idx  = 0
best_params_overall = None
best_model_obj   = None

for cfg_idx, config in enumerate(CNN_CONFIGS):
    run_name = f"cnn_config_{cfg_idx+1}_f{config['filters1']}_{config['filters2']}"
    print(f"\n{'='*50}")
    print(f"Config {cfg_idx+1}/5: {config}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type":   "JAX_Flax_1D_CNN",
            "filters1":     config["filters1"],
            "filters2":     config["filters2"],
            "dense_units":  config["dense"],
            "lr":           config["lr"],
            "batch_size":   config["batch_size"],
            "epochs":       config["epochs"],
            "seq_len":      SEQ_LEN,
            "n_features":   N_FEATURES,
            "optimizer":    "adam",
            "pooling":      "global_average",
            "class_weight": 10.0,
        })

        # Init model
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        model  = TelemetryCNN(filters1=config["filters1"],
                              filters2=config["filters2"],
                              dense=config["dense"])
        sample = jnp.array(X_train[:1])
        params = model.init(init_rng, sample, training=False)
        tx     = optax.adam(config["lr"])
        state  = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx)

        best_f1, best_params = 0.0, params

        for epoch in range(config["epochs"]):
            # Train
            perm   = np.random.permutation(len(X_train))
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]
            t_losses, t_accs = [], []
            for i in range(0, len(X_train), config["batch_size"]):
                xb = jnp.array(X_shuf[i:i+config["batch_size"]])
                yb = jnp.array(y_shuf[i:i+config["batch_size"]])
                rng, drng = jax.random.split(rng)
                state, loss, acc = train_step(state, xb, yb, drng)
                t_losses.append(float(loss))
                t_accs.append(float(acc))

            # Validate
            v_preds, v_labels = [], []
            for i in range(0, len(X_val), config["batch_size"]*2):
                xb = jnp.array(X_val[i:i+config["batch_size"]*2])
                yb = jnp.array(y_val[i:i+config["batch_size"]*2])
                _, _, preds = eval_step(state, xb, yb)
                v_preds.extend(np.array(preds))
                v_labels.extend(np.array(yb))

            val_f1 = f1_score(v_labels, v_preds, zero_division=0)
            print(f"  Epoch {epoch+1}/{config['epochs']} | "
                  f"loss={np.mean(t_losses):.4f} "
                  f"acc={np.mean(t_accs):.3f} | "
                  f"val_f1={val_f1:.3f}")

            mlflow.log_metrics({
                "train_loss": float(np.mean(t_losses)),
                "train_acc":  float(np.mean(t_accs)),
                "val_f1":     val_f1,
            }, step=epoch)

            if val_f1 > best_f1:
                best_f1     = val_f1
                best_params = state.params

        # Final eval
        all_preds, all_labels = [], []
        for i in range(0, len(X_val), config["batch_size"]*2):
            xb = jnp.array(X_val[i:i+config["batch_size"]*2])
            yb = jnp.array(y_val[i:i+config["batch_size"]*2])
            tmp_state = train_state.TrainState.create(
                apply_fn=model.apply, params=best_params, tx=optax.adam(config["lr"]))
            _, _, preds = eval_step(tmp_state, xb, yb)
            all_preds.extend(np.array(preds))
            all_labels.extend(np.array(yb))

        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)

        mlflow.log_metrics({
            "final_accuracy":  acc,
            "final_precision": prec,
            "final_recall":    rec,
            "final_f1":        f1,
        })

        print(f"  ✅ Config {cfg_idx+1} → Acc:{acc*100:.1f}% P:{prec*100:.1f}% R:{rec*100:.1f}% F1:{f1*100:.1f}%")

        if best_f1 > best_f1_overall:
            best_f1_overall     = best_f1
            best_params_overall = best_params
            best_model_obj      = model
            best_config_idx     = cfg_idx


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("JAX/FLAX CNN TRAINING COMPLETE")
print("="*60)
print(f"Best config:  {best_config_idx+1}")
print(f"Best val F1:  {best_f1_overall:.4f}")
print(f"MLflow exp:   /Users/{current_user}/vehicle-telemetry-cnn-jax")
print("="*60)
print("✅ CNN training complete! Check MLflow for all 5 runs.")
