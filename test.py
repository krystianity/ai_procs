import pandas as pd
from aiprocs import procs

pd.set_option("display.max_rows", 2)

# df must be train only (split train and validation beforehand)
df = pd.read_csv("ds/heart.csv")

# describe your dataset (dataframe) columns
options = procs.create_options(
    ["FillMissing", "Normalize", "Categorize"], ["age", "thal"], "target"
)

# generate metadata from for dataframe
metadata = procs.analyse_df(df, options)
procs.store_metadata(metadata, "./metadata.json")
# metadata = procs.load_metadata("./metadata.json")
print("Original:\n", df)

# use metadata to optimize your dataframe for ml
df = procs.prepare_df(df, metadata)
print("Prepared:\n", df)
