import pandas as pd
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Load the sample data (tab-separated)
data_path = script_dir / "sample_data.txt"
df = pd.read_csv(data_path, sep="\t")

if __name__ == "__main__":
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
