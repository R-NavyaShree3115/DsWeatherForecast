import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv(r"C:\Users\RNavy\OneDrive\DsWeatherForecast\GlobalWeatherRepository.csv")
features = ["humidity", "wind_kph", "pressure_mb", "precip_mm"]
df = df.dropna(subset=features + ["temperature_celsius"])
X = df[features]
y = df["temperature_celsius"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
sample_input = pd.DataFrame({
    "humidity": [70],
    "wind_kph": [15],
    "pressure_mb": [1015],

    "precip_mm": [0.5]
})
predicted_temperature = model.predict(sample_input)
print(f"Predicted Temperature: {predicted_temperature[0]}")