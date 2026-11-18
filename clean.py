import pandas as pd
file_path = r"C:\Users\RNavy\OneDrive\DsWeatherForecast\GlobalWeatherRepository.csv"
df = pd.read_csv(file_path)
df = df.drop_duplicates()
df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
numbers = df.select_dtypes(include=["number"]).columns
df[numbers] = df[numbers].fillna(df[numbers].median())
df=df.fillna("Unknown")
df.to_csv(file_path, index=False)
print("Data cleaned and saved successfully.")