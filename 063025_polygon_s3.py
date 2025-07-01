import pandas as pd
import subprocess
from pathlib import Path
from datetime import datetime

# Define date range: from Jan 1, 2025 to last business day before today
start_date = "2024-01-01"
today = pd.Timestamp.today().normalize()
bizdays = pd.date_range(start=start_date, end=today - pd.Timedelta(days=1), freq="B")
end_date = '2024-12-31'  # Default end date
if not bizdays.empty:
    end_date = bizdays[-1].strftime("%Y-%m-%d")
else:
    raise ValueError("No business days before today")

# Config
endpoint = "https://files.polygon.io"
bucket_root = "s3://flatfiles/us_options_opra/day_aggs_v1/2024"
output_dir = Path("./polygon_data_2024")
output_dir.mkdir(parents=True, exist_ok=True)

for date in bizdays:
    date_str = date.strftime("%Y-%m-%d")
    month_str = date.strftime("%m")
    s3_path = f"{bucket_root}/{month_str}/{date_str}.csv.gz"
    local_path = output_dir / f"{date_str}.csv.gz"

    print(f"Checking {s3_path}...")
    
    ls_result = subprocess.run(
        ["aws", "s3", "ls", s3_path, "--endpoint-url", endpoint],
        capture_output=True, text=True
    )

    if ls_result.returncode == 0:
        print(f"Downloading {s3_path}...")
        cp_result = subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_path), "--endpoint-url", endpoint],
            capture_output=True, text=True
        )
        if cp_result.returncode != 0:
            print(f"Download error: {cp_result.stderr.strip()}")
    else:
        print(f"Not found or forbidden: {s3_path}")
