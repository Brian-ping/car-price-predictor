# quick_test.py
import pandas as pd
from config import *

print("=== Quick Test of Cardekho Dataset ===")

# Load raw data
df_raw = pd.read_csv(RAW_DATA_PATH, nrows=10)
print(f"1. Raw data sample (10 rows):")
print(f"   Shape: {df_raw.shape}")
print(f"   Columns: {list(df_raw.columns)}")
print(f"   First car: {df_raw.iloc[0]['name']}")
print(f"   Price: ₹{df_raw.iloc[0]['selling_price']:,}")
print(f"   Year: {df_raw.iloc[0]['year']}")
print(f"   Fuel: {df_raw.iloc[0]['fuel']}")
print()

# Check if processed data exists
try:
    df_proc = pd.read_csv(PROCESSED_DATA_PATH, nrows=5)
    print(f"2. Processed data sample (5 rows):")
    print(f"   Shape: {df_proc.shape}")
    print(f"   Available columns: {list(df_proc.columns)}")
    print(f"   First car: {df_proc.iloc[0]['car_name']}")
    print(f"   Brand: {df_proc.iloc[0]['brand']}")
    print(f"   Car age: {df_proc.iloc[0]['car_age']} years")
    print(f"   Owner numeric: {df_proc.iloc[0]['owner_numeric']}")
except FileNotFoundError:
    print(f"2. Processed data not found at: {PROCESSED_DATA_PATH}")
    print("   Run prepare_cardekho_data.py first")

print()
print("=== Dataset Statistics ===")
if 'df_proc' in locals():
    print(f"Total cars: {len(df_proc):,}")
    print(f"Price range: ₹{df_proc['selling_price'].min():,.0f} - ₹{df_proc['selling_price'].max():,.0f}")
    print(f"Average price: ₹{df_proc['selling_price'].mean():,.0f}")
    print(f"Brands: {df_proc['brand'].nunique()}")
    print(f"Fuel types: {df_proc['fuel_type'].unique()}")
