import json
import pandas as pd
from enum import Enum

__version__ = "0.1.0"


class COL_TYPES(Enum):
    NORMALIZED = "NORMALIZED"
    CATEGORIZED = "CATEGORIZED"


def _create_metadata(options):
    return {"version": __version__, "options": options, "columns": {}}


def _standardize_vec(vec, mean, std):
    return vec.sub(mean).div(std)


def _analyse_fill_missing(df, options, metadata):
    return


def _analyse_standardization(df, options, metadata):
    columns = list(df.columns.values)
    for column in columns:
        if column not in options["cat_names"]:
            mean = df[column].mean()
            std = df[column].std()
            metadata["columns"][column] = {
                "type": COL_TYPES.NORMALIZED.value,
                "mean": mean,
                "std": std,
            }


def _analyse_categorization(df, options, metadata):
    for cat in options["cat_names"]:
        metadata["columns"][cat] = {"type": COL_TYPES.CATEGORIZED.value}


def store_metadata(metadata, filepath):
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4)
    return


def analyse_df(df, options):

    metadata = _create_metadata(options)
    for proc in options["procs"]:
        if proc == "FillMissing":
            _analyse_fill_missing(df, options, metadata)
        elif proc == "Normalize":
            _analyse_standardization(df, options, metadata)
        elif proc == "Categorize":
            _analyse_categorization(df, options, metadata)
        else:
            raise ValueError("Unsupported proc " + proc)

    return metadata


def prepare_df(df, metadata):
    columns = list(df.columns.values)
    for column in columns:

        if column not in metadata["columns"]:
            raise ValueError("Unsupported column in df " + column)

        info = metadata["columns"][column]
        if info["type"] == COL_TYPES.NORMALIZED.value:
            df[column] = _standardize_vec(df[column], info["mean"], info["std"])

    return df
