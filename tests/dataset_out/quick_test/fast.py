import pandas as pd
import os

# Define the exact path to your Parquet file
FILE_PATH = r"tests\dataset_out\quick_test\benchmark_metrics.parquet"

def inspect_parquet(file_path):
    """
    Reads and comprehensively displays the contents of the benchmark Parquet file.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found at: {file_path}")
        return

    # Load the dataset
    print(f"[*] Loading data from: {file_path}")
    df = pd.read_parquet(file_path)

    # 1. Basic Dataset Info
    print("\n" + "="*50)
    print("--- DATASET SHAPE & BASIC INFO ---")
    print("="*50)
    print(f"Total Rows (Experiments): {len(df)}")
    print(f"Total Columns (Metrics/Config): {len(df.columns)}")
    print(f"Processors tested: {df['processor'].unique()}")
    print(f"WPE usage tested: {df['use_wpe'].unique()}")

    # 2. Configure Pandas to show ALL columns and rows without truncating
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    # 3. Filter and display specific metric groups
    # Let's focus on PESQ and STOI since those are the perceptual metrics in question
    metrics_to_inspect = ["PESQ", "STOI"]

    for metric in metrics_to_inspect:
        print("\n" + "="*80)
        print(f"--- DETAILED BREAKDOWN: {metric} ---")
        print("="*80)
        
        # Gather all columns related to this specific metric
        # We include the configuration columns first for context
        config_cols = ["processor", "use_wpe", "rt60", "isir_db"]
        
        # Absolute scores
        abs_cols = [c for c in df.columns if metric in c and "Delta" not in c]
        
        # Delta scores (Improvements)
        delta_cols = [c for c in df.columns if metric in c and "Delta" in c]
        
        # Combine them safely, ensuring they actually exist in the dataframe
        cols_to_show = config_cols + abs_cols + delta_cols
        cols_to_show = [c for c in cols_to_show if c in df.columns]

        if len(cols_to_show) > len(config_cols):
            # Sort values to make it easier to read (e.g., group by processor)
            display_df = df[cols_to_show].sort_values(by=["processor", "use_wpe"])
            print(display_df.to_string(index=False))
        else:
            print(f"No columns found for metric: {metric}")

    # 4. Optional: Show all column names just in case you need to find a specific one
    print("\n" + "="*50)
    print("--- ALL AVAILABLE COLUMNS ---")
    print("="*50)
    for col in df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    inspect_parquet(FILE_PATH)