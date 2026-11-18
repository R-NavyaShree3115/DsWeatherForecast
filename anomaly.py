import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest

df = pd.read_csv(r"C:\Users\RNavy\OneDrive\DsWeatherForecast\GlobalWeatherRepository.csv")
features = ["humidity", "wind_kph", "pressure_mb", "precip_mm"]
df = df[features].dropna()

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(df[features])

anomaly_labels = model.predict(df[features])
df["anomaly"] = anomaly_labels

print("Total observations:", len(df))
print("Normal observations:", (df["anomaly"] == 1).sum())
print("Anomalies detected:", (df["anomaly"] == -1).sum())

print("\nUnusual weather observations:")
print(df[df["anomaly"] == -1].head(10))