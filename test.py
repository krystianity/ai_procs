import pandas as pd
from ai_procs import procs

pd.set_option("display.max_rows", 2)

options = {
    "procs": ["FillMissing", "Normalize", "Categorize"],
    "cat_names": ["age", "thal"],
    "dep_var": "target",
}

# df must be train only (split before)
df = pd.read_csv("ds/heart.csv")

metadata = procs.analyse_df(df, options)
procs.store_metadata(metadata, "./metadata.json")
print("Original:", df)
df = procs.prepare_df(df, metadata)
print("Prepared:", df)
