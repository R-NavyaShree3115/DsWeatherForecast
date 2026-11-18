import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\RNavy\OneDrive\DsWeatherForecast\GlobalWeatherRepository.csv")
plt.figure(figsize=(8, 5))
sns.histplot(df["temperature_celsius"], bins=30, kde=True)
plt.title("Temperature Distribution")
plt.xlabel("Temperature (celsius)")
plt.ylabel("Frequency")
plt.savefig("temperature_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="humidity",
    y="temperature_celsius",
)
plt.title("Humidity vs Temperature")
plt.savefig("humidity_vs_temperature.png")
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(
    df[["temperature_celsius", "humidity", "wind_kph", "pressure_mb","precip_mm"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

print("EDA completed successfully")