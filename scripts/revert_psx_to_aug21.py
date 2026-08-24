import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PSX_DIR = ROOT / "data-new" / "psx-data"

def revert_psx():
    if not PSX_DIR.exists():
        print(f"Directory {PSX_DIR} does not exist.")
        return

    cutoff = pd.Timestamp("2026-08-24")
    modified_count = 0

    for csv_file in PSX_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, index_col="Date", parse_dates=True)
            if df.empty:
                continue
            
            # Filter out any rows on or after 2026-08-24
            keep = df.index < cutoff
            if (~keep).sum() > 0:
                df_clean = df[keep]
                df_clean.to_csv(csv_file)
                modified_count += 1
                print(f"Trimmed {csv_file.name}: new max date = {df_clean.index.max().date()}")
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    print(f"Revert complete: modified {modified_count} PSX CSV files.")

if __name__ == "__main__":
    revert_psx()
