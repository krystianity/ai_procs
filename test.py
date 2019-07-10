import pandas as pd
from aiprocs import aip

pd.set_option("display.max_rows", 2)

# df must be train only (split train and validation beforehand)
df = pd.read_csv("ds/heart.csv")

# describe your dataset (dataframe) columns
options = aip.create_options(
    ["FillMissing", "Normalize", "Categorify"],
    ["age", "thal"],
    "target",
    aip.MODEL_TYPES.PYTORCH.value,
)

# generate metadata from for dataframe
metadata = aip.analyse_df(df, options)
aip.store_metadata(metadata, "./metadata.json")
# metadata = aip.load_metadata("./metadata.json")
print("Original:\n", df)

# use metadata to optimize your dataframe for ml
df = aip.prepare_df(df, metadata)
print("Prepared:\n", df)
