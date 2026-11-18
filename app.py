
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="WeatherScope AI", layout="wide")

df = pd.read_csv(r"C:\Users\RNavy\OneDrive\DsWeatherForecast\GlobalWeatherRepository.csv")
df = df.drop_duplicates()

st.title("WeatherScope AI")
st.write("Global Weather Analysis and Prediction")

country = st.selectbox(
    "Select Country",
    ["All"] + sorted(df["country"].dropna().unique().tolist())
)

if country != "All":
    df = df[df["country"] == country]

st.subheader("Temperature Distribution")

fig = px.histogram(
    df,
    x="temperature_celsius",
    nbins=30,
    title="Global Temperature Distribution"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Humidity vs Temperature in Different Countries")

fig = px.scatter(
    df,
    x="humidity",
    y="temperature_celsius",
    color="country",
    hover_name="country",
    title="Humidity vs Temperature by Country",
    labels={
        "humidity": "Humidity (%)",
        "temperature_celsius": "Temperature (°C)"
    }
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Correlation Matrix")

features = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm"
]

corr = df[features].corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Weather Feature Correlation"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Temperature Prediction")

features = [
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm",
    "uv_index",
    "latitude",
    "longitude"
]

model_data = df[features + ["temperature_celsius"]].dropna()

if len(model_data) > 10:
    X = model_data[features]
    y = model_data["temperature_celsius"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mean_absolute_error(y_test, predictions):.2f}")
    c2.metric("R2 Score", f"{r2_score(y_test, predictions):.2f}")
else:
    st.write("Not enough records for prediction.")

st.subheader("Anomaly Detection")

features = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm"
]

anomaly_data = df[features].dropna().copy()

if len(anomaly_data) > 10:
    detector = IsolationForest(contamination=0.05, random_state=42)

    anomaly_data["Status"] = detector.fit_predict(anomaly_data[features])
    anomaly_data["Status"] = anomaly_data["Status"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    st.metric(
        "Anomalies Detected",
        (anomaly_data["Status"] == "Anomaly").sum()
    )

    st.dataframe(
        anomaly_data[anomaly_data["Status"] == "Anomaly"],
        use_container_width=True
    )
else:
    st.write("Not enough records for anomaly detection.")
