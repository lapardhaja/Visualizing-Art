import pandas as pd
from pathlib import Path

# Setting static file paths (used for new folder structure)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Grabbing the proper csv files
reduced_csv = DATA_DIR / "summary_combined_reduced.csv"
met_csv = DATA_DIR / "met_paintings.csv"
output_csv = DATA_DIR / "summary_combined_reduced_with_links.csv"

# Loading CSV's
df1 = pd.read_csv(reduced_csv)
df2 = pd.read_csv(met_csv)

# Merge only the Link Resource for query_object_id
df1 = df1.merge(
    df2[['Object ID', 'Link Resource']].rename(
        columns={'Object ID': 'query_object_id', 'Link Resource': 'query_link'}
    ),
    on='query_object_id',
    how='left'
)

# Merge only the Link Resource for similar_object_id
df1 = df1.merge(
    df2[['Object ID', 'Link Resource']].rename(
        columns={'Object ID': 'similar_object_id', 'Link Resource': 'similar_link'}
    ),
    on='similar_object_id',
    how='left'
)

# Save the final file
df1.to_csv(output_csv, index=False)
print(f"Wrote {output_csv}")