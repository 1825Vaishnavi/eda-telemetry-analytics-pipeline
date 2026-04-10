import numpy as np
import pandas as pd
from scipy import stats
import os

os.makedirs("reports", exist_ok=True)

SENSOR_COLS = ["speed_kmh","battery_pct","temp_motor_c","voltage_v","current_a"]

np.random.seed(42)
ref = pd.DataFrame({c: np.random.normal(0,1,5000) for c in SENSOR_COLS})
cur = pd.DataFrame({c: np.random.normal(0.5,1.2,2000) for c in SENSOR_COLS})

print("🔍 Running drift detection...")
results = {}
for col in SENSOR_COLS:
    stat, p = stats.ks_2samp(ref[col], cur[col])
    drifted = p < 0.05
    results[col] = {"ks_stat": round(float(stat),4), "p_value": round(float(p),4), "drift": drifted}
    status = "🔴 DRIFT" if drifted else "✅ OK"
    print(f"  {status}: {col} (p={p:.4f})")

drifted_cols = [c for c,v in results.items() if v["drift"]]
pd.DataFrame(results).T.to_csv("reports/drift_report.csv")
print(f"\n✅ {len(drifted_cols)}/{len(SENSOR_COLS)} columns drifted")
print("✅ Report saved → reports/drift_report.csv")
