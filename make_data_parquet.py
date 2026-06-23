"""Convert the 104MB violation CSV into a compact parquet with only the
columns the app needs, so it fits in GitHub and deploys on Streamlit Cloud."""
import pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8")

CSV = "jan to may police violation_anonymized791b166.csv"
OUT = "data.parquet"

KEEP = ["id", "latitude", "longitude", "vehicle_type", "violation_type",
        "created_datetime", "police_station", "data_sent_to_scita", "junction_name"]

print("Reading CSV...")
df = pd.read_csv(CSV, usecols=KEEP, low_memory=False)
print(f"  {len(df):,} rows, {len(df.columns)} cols")

# keep created_datetime as string for portable re-parse
df["created_datetime"] = df["created_datetime"].astype(str)

df.to_parquet(OUT, index=False, compression="snappy")
import os
mb = os.path.getsize(OUT) / 1e6
print(f"Saved {OUT}: {mb:.1f} MB")
