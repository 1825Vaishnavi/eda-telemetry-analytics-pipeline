import os, time, logging, numpy as np
from typing import List, Optional
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SENSOR_COLS = ["speed_kmh","battery_pct","temp_motor_c","temp_battery_c","regen_brake_kw","accel_x","accel_y","accel_z","voltage_v","current_a"]
N_FEATURES = len(SENSOR_COLS)
SEQ_LEN = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm_best.pt")

class TelemetryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, 2))
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)

class ModelStore:
    model = None
    scaler_mean = None
    scaler_std = None
    loaded = False
    predict_count = 0
    total_latency = 0.0

model_store = ModelStore()

def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            m = TelemetryLSTM(N_FEATURES, 64, 2, 0.2).to(DEVICE)
            m.load_state_dict(checkpoint["model_state_dict"])
            m.eval()
            model_store.model = m
            model_store.scaler_mean = np.array(checkpoint.get("scaler_mean", [0]*N_FEATURES))
            model_store.scaler_std = np.array(checkpoint.get("scaler_std", [1]*N_FEATURES))
        else:
            m = TelemetryLSTM(N_FEATURES, 64, 2, 0.2).to(DEVICE)
            m.eval()
            model_store.model = m
            model_store.scaler_mean = np.zeros(N_FEATURES)
            model_store.scaler_std = np.ones(N_FEATURES)
        model_store.loaded = True
        log.info("Model loaded!")
    except Exception as e:
        log.error(f"Model load failed: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="Vehicle Telemetry Anomaly Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SensorReading(BaseModel):
    vehicle_id: str = Field(..., example="VH_00001")
    speed_kmh: float = Field(..., example=85.5)
    battery_pct: float = Field(..., example=72.3)
    temp_motor_c: float = Field(..., example=78.1)
    temp_battery_c: float = Field(..., example=32.4)
    regen_brake_kw: float = Field(..., example=4.2)
    accel_x: float = Field(..., example=0.3)
    accel_y: float = Field(..., example=-0.1)
    accel_z: float = Field(..., example=9.8)
    voltage_v: float = Field(..., example=398.5)
    current_a: float = Field(..., example=105.2)

class PredictResponse(BaseModel):
    vehicle_id: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    label: str
    latency_ms: float

class BatchRequest(BaseModel):
    readings: List[SensorReading]

class BatchResponse(BaseModel):
    total: int
    anomalies_found: int
    anomaly_rate: float
    results: List[PredictResponse]
    latency_ms: float

def reading_to_vector(r):
    return np.array([r.speed_kmh, r.battery_pct, r.temp_motor_c, r.temp_battery_c, r.regen_brake_kw, r.accel_x, r.accel_y, r.accel_z, r.voltage_v, r.current_a], dtype=np.float32)

def normalize(v):
    return (v - model_store.scaler_mean) / (model_store.scaler_std + 1e-8)

def build_sequence(reading):
    current = normalize(reading_to_vector(reading))
    seq = np.tile(current, (SEQ_LEN, 1))
    return seq.reshape(1, SEQ_LEN, N_FEATURES)

def run_lstm(seq_np):
    t = torch.tensor(seq_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model_store.model(t)
        probs = torch.softmax(logits, dim=1)
    anomaly_prob = float(probs[0, 1])
    return anomaly_prob > 0.5, anomaly_prob, max(anomaly_prob, 1 - anomaly_prob)

@app.get("/health")
def health():
    return {"status": "healthy" if model_store.loaded else "loading", "model_loaded": model_store.loaded, "device": DEVICE}

@app.get("/metrics")
def metrics():
    avg = model_store.total_latency / model_store.predict_count if model_store.predict_count > 0 else 0.0
    return {"total_predictions": model_store.predict_count, "avg_latency_ms": round(avg, 2)}

@app.post("/predict", response_model=PredictResponse)
def predict(reading: SensorReading):
    if not model_store.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    seq = build_sequence(reading)
    is_anomaly, score, confidence = run_lstm(seq)
    latency_ms = (time.perf_counter() - t0) * 1000
    model_store.predict_count += 1
    model_store.total_latency += latency_ms
    return PredictResponse(vehicle_id=reading.vehicle_id, is_anomaly=is_anomaly, anomaly_score=round(score,4), confidence=round(confidence,4), label="ANOMALY" if is_anomaly else "NORMAL", latency_ms=round(latency_ms,2))

@app.post("/batch_predict", response_model=BatchResponse)
def batch_predict(batch: BatchRequest):
    if not model_store.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    seqs = np.vstack([build_sequence(r) for r in batch.readings])
    t_in = torch.tensor(seqs, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model_store.model(t_in)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000
    results = []
    for i, reading in enumerate(batch.readings):
        anomaly_prob = float(probs[i, 1])
        is_anomaly = anomaly_prob > 0.5
        results.append(PredictResponse(vehicle_id=reading.vehicle_id, is_anomaly=is_anomaly, anomaly_score=round(anomaly_prob,4), confidence=round(max(anomaly_prob,1-anomaly_prob),4), label="ANOMALY" if is_anomaly else "NORMAL", latency_ms=round(latency_ms/len(batch.readings),2)))
    anomalies = sum(1 for r in results if r.is_anomaly)
    model_store.predict_count += len(batch.readings)
    model_store.total_latency += latency_ms
    return BatchResponse(total=len(batch.readings), anomalies_found=anomalies, anomaly_rate=round(anomalies/len(batch.readings),4), results=results, latency_ms=round(latency_ms,2))

@app.get("/")
def root():
    return {"name": "Vehicle Telemetry Anomaly Detection API", "version": "1.0.0", "docs": "/docs"}
