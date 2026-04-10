"""
cnn_model.py
JAX/Flax 1D CNN — Signal Shape Classification
Classifies vehicle sensor signal patterns as normal or anomalous.

Usage:
    python src/cnn_model.py
"""

import os
import logging
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SENSOR_COLS = [
    "speed_kmh", "battery_pct", "temp_motor_c", "temp_battery_c",
    "regen_brake_kw", "accel_x", "accel_y", "accel_z",
    "voltage_v", "current_a"
]
SEQ_LEN    = 30
N_FEATURES = len(SENSOR_COLS)
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 5 CNN configs for MLflow
CNN_CONFIGS = [
    {"filters1": 16,  "filters2": 32,  "dense": 32,  "lr": 1e-3, "batch_size": 64,  "epochs": 5},
    {"filters1": 32,  "filters2": 64,  "dense": 64,  "lr": 1e-3, "batch_size": 64,  "epochs": 5},
    {"filters1": 64,  "filters2": 128, "dense": 64,  "lr": 5e-4, "batch_size": 128, "epochs": 5},
    {"filters1": 32,  "filters2": 64,  "dense": 128, "lr": 1e-3, "batch_size": 32,  "epochs": 5},
    {"filters1": 64,  "filters2": 128, "dense": 128, "lr": 1e-4, "batch_size": 128, "epochs": 5},
]


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def generate_telemetry(n_records=30_000):
    log.info(f"Generating {n_records:,} telemetry records...")
    np.random.seed(0)
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
    df     = pd.DataFrame(data)
    labels = np.zeros(n_records, dtype=int)
    idx    = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
    df.loc[idx, "temp_motor_c"] += np.random.uniform(40, 80, len(idx))
    df.loc[idx, "battery_pct"]  -= np.random.uniform(30, 60, len(idx))
    labels[idx] = 1
    df["label"] = labels
    return df


def make_sequences(X, y, seq_len=SEQ_LEN):
    seqs, lbls = [], []
    for i in range(len(X) - seq_len):
        seqs.append(X[i:i+seq_len])
        lbls.append(y[i+seq_len])
    return np.array(seqs, dtype=np.float32), np.array(lbls, dtype=np.int32)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class TelemetryCNN(nn.Module):
    filters1: int
    filters2: int
    dense:    int

    @nn.compact
    def __call__(self, x, training: bool = False):
        # x: (batch, seq_len, features)
        x = nn.Conv(features=self.filters1, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.filters1, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))

        x = nn.Conv(features=self.filters2, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.filters2, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        x = x.mean(axis=1)                           # global average pool

        x = nn.Dense(self.dense)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.3)(x, deterministic=not training)
        x = nn.Dense(2)(x)
        return x


# ─────────────────────────────────────────────
# TRAIN STATE
# ─────────────────────────────────────────────
def create_train_state(rng, config, sample_batch):
    model = TelemetryCNN(
        filters1=config["filters1"],
        filters2=config["filters2"],
        dense=config["dense"],
    )
    params = model.init(rng, sample_batch, training=False)
    tx     = optax.adam(config["lr"])
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), model


# ─────────────────────────────────────────────
# LOSS + METRICS
# ─────────────────────────────────────────────
def cross_entropy_loss(logits, labels):
    # Class weights: 1.0 normal, 10.0 anomaly
    weights     = jnp.where(labels == 1, 10.0, 1.0)
    one_hot     = jax.nn.one_hot(labels, num_classes=2)
    log_softmax = jax.nn.log_softmax(logits)
    loss        = -jnp.sum(one_hot * log_softmax, axis=-1)
    return jnp.mean(loss * weights)

@jax.jit
def train_step(state, batch_x, batch_y, dropout_rng):
    def loss_fn(params):
        logits = state.apply_fn(
            params, batch_x, training=True,
            rngs={"dropout": dropout_rng}
        )
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


# ─────────────────────────────────────────────
# TRAIN ONE CONFIG
# ─────────────────────────────────────────────
def train_config(config, X_train, y_train, X_val, y_val, run_name):
    rng        = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    sample = jnp.array(X_train[:1])
    state, model = create_train_state(init_rng, config, sample)

    best_f1, best_params = 0.0, state.params
    n_train = len(X_train)

    for epoch in range(config["epochs"]):
        # Shuffle
        perm    = np.random.permutation(n_train)
        X_shuf  = X_train[perm]
        y_shuf  = y_train[perm]

        # Train batches
        train_losses, train_accs = [], []
        for i in range(0, n_train, config["batch_size"]):
            xb = jnp.array(X_shuf[i:i+config["batch_size"]])
            yb = jnp.array(y_shuf[i:i+config["batch_size"]])
            rng, dropout_rng = jax.random.split(rng)
            state, loss, acc = train_step(state, xb, yb, dropout_rng)
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        # Validation
        val_preds, val_labels = [], []
        val_losses = []
        for i in range(0, len(X_val), config["batch_size"] * 2):
            xb = jnp.array(X_val[i:i+config["batch_size"]*2])
            yb = jnp.array(y_val[i:i+config["batch_size"]*2])
            loss, acc, preds = eval_step(state, xb, yb)
            val_losses.append(float(loss))
            val_preds.extend(np.array(preds))
            val_labels.extend(np.array(yb))

        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        log.info(
            f"[{run_name}] Epoch {epoch+1}/{config['epochs']} | "
            f"train_loss={np.mean(train_losses):.4f} "
            f"train_acc={np.mean(train_accs):.3f} | "
            f"val_loss={np.mean(val_losses):.4f} "
            f"val_f1={val_f1:.3f}"
        )

        mlflow.log_metrics({
            "train_loss":  float(np.mean(train_losses)),
            "train_acc":   float(np.mean(train_accs)),
            "val_loss":    float(np.mean(val_losses)),
            "val_f1":      val_f1,
        }, step=epoch)

        if val_f1 > best_f1:
            best_f1     = val_f1
            best_params = state.params

    return best_params, best_f1, model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info(f"JAX backend: {jax.default_backend()}")

    df       = generate_telemetry(30_000)
    X        = df[SENSOR_COLS].values
    y        = df["label"].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = make_sequences(X_scaled, y)
    split         = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("vehicle-telemetry-cnn")

    best_f1_overall, best_params_overall, best_model_obj = 0.0, None, None
    best_config_idx = 0

    for i, config in enumerate(CNN_CONFIGS):
        run_name = f"cnn_config_{i+1}_f{config['filters1']}_{config['filters2']}"
        log.info(f"\n{'='*60}")
        log.info(f"CNN Config {i+1}/5: {config}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_type":  "1D_CNN",
                "filters1":    config["filters1"],
                "filters2":    config["filters2"],
                "dense_units": config["dense"],
                "lr":          config["lr"],
                "batch_size":  config["batch_size"],
                "epochs":      config["epochs"],
                "seq_len":     SEQ_LEN,
                "n_features":  N_FEATURES,
                "optimizer":   "adam",
                "pooling":     "global_average",
                "class_weight_anomaly": 10.0,
            })

            best_params, best_f1, model_obj = train_config(
                config, X_train, y_train, X_val, y_val, run_name
            )

            # Final eval
            all_preds, all_labels = [], []
            for i2 in range(0, len(X_val), config["batch_size"]*2):
                xb = jnp.array(X_val[i2:i2+config["batch_size"]*2])
                yb = jnp.array(y_val[i2:i2+config["batch_size"]*2])
                _, _, preds = eval_step(
                    train_state.TrainState.create(
                        apply_fn=model_obj.apply,
                        params=best_params,
                        tx=optax.adam(config["lr"])
                    ), xb, yb
                )
                all_preds.extend(np.array(preds))
                all_labels.extend(np.array(yb))

            from sklearn.metrics import accuracy_score, precision_score, recall_score
            acc  = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec  = recall_score(all_labels, all_preds, zero_division=0)
            f1   = f1_score(all_labels, all_preds, zero_division=0)

            mlflow.log_metrics({
                "final_accuracy":  acc,
                "final_precision": prec,
                "final_recall":    rec,
                "final_f1":        f1,
                "best_val_f1":     best_f1,
            })

            log.info(f"Config {i+1} — Acc:{acc*100:.1f}% P:{prec*100:.1f}% R:{rec*100:.1f}% F1:{f1*100:.1f}%")

            if best_f1 > best_f1_overall:
                best_f1_overall      = best_f1
                best_params_overall  = best_params
                best_model_obj       = model_obj
                best_config_idx      = i

    # Save best model params as numpy
    best_cfg = CNN_CONFIGS[best_config_idx]
    np.save(os.path.join(MODEL_DIR, "cnn_best_params.npy"),
            jax.tree_util.tree_map(np.array, best_params_overall))
    np.save(os.path.join(MODEL_DIR, "cnn_config.npy"), best_cfg)

    log.info(f"\n✅ Best CNN: config_{best_config_idx+1}, F1={best_f1_overall:.4f}")
    log.info(f"✅ Params saved → models/cnn_best_params.npy")
    log.info("✅ CNN training complete! Check MLflow at http://localhost:5000")


if __name__ == "__main__":
    main()
