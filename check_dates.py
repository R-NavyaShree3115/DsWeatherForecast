
import pandas as pd

df = pd.read_csv(r"C:\Users\RNavy\OneDrive\DsWeatherForecast\GlobalWeatherRepository.csv")

df["last_updated"] = pd.to_datetime(df["last_updated"])

print("Total records:", len(df))
print("Unique dates:", df["last_updated"].dt.date.nunique())
print("Date range:", df["last_updated"].min(), "to", df["last_updated"].max())
