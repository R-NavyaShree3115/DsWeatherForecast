import pandas as pd
file_path = r"C:\Users\RNavy\OneDrive\DsWeatherForecast\GlobalWeatherRepository.csv"
try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    data.columns = (
        data.columns.str.strip().str.replace("\ufeff", "", regex=False)
    )
    print("Column names cleaned.")
    print("Dataset shape:", data.shape)
    print("\nColumns:\n", data.columns.tolist())
    print("\nFirst 5 rows:\n", data.head())
    print("\nMissing values:\n", data.isnull().sum())
    print("\nDate range:")
    print(data["last_updated"].min(), "to", data["last_updated"].max())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")

    