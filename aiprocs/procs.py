import json
import pandas as pd
import numpy as np
from enum import Enum

__version__ = "0.1.0"


class COL_TYPES(Enum):
    NORMALIZED = "NORMALIZED"
    CATEGORIZED = "CATEGORIZED"
    DEPVAR = "DEPVAR"


def _create_metadata(options):
    return {"version": __version__, "options": options, "columns": {}}


def _standardize_vec(vec, mean, std):
    return vec.sub(mean).div(std)


def _map_vec(vec, hashmap):
    return vec.map(hashmap)


def _analyse_fill_missing(df, options, metadata):
    return


def _analyse_standardization(df, options, metadata):
    columns = list(df.columns.values)
    ignore_list = options["cat_names"].copy()
    ignore_list.append(options["dep_var"])
    for column in columns:
        if column not in ignore_list:
            mean = df[column].mean()
            std = df[column].std()
            metadata["columns"][column] = {
                "type": COL_TYPES.NORMALIZED.value,
                "mean": mean,
                "std": std,
            }


def _analyse_categorization(df, options, metadata):
    for cat in options["cat_names"]:

        u_values = df[cat].unique().tolist()
        values = {}
        for i, val in enumerate(u_values):
            values[val] = i

        metadata["columns"][cat] = {
            "type": COL_TYPES.CATEGORIZED.value,
            "values": values,
        }


def _validate_options(options):

    if "procs" not in options:
        raise ValueError("Missing procs key in options dict")

    if "cat_names" not in options:
        raise ValueError("Missing cat_names key in options")

    if "dep_var" not in options:
        raise ValueError("Missing dep_var key in options dict")

    return


def split_df(df, valid_frac=0.2, test_frac=0.0):
    df_len = len(df)
    random_idx = np.random.permutation(df_len)
    df = df.iloc[random_idx]
    valid_cut = int(valid_frac * df_len)
    test_cut = valid_cut + int(test_frac * df_len)
    valid_df = df.iloc[:valid_cut]
    test_df = df.iloc[valid_cut:test_cut]
    train_df = df.iloc[test_cut:]
    return train_df, valid_df, test_df


def create_options(procs, cat_names, dep_var):
    return {"procs": procs, "cat_names": cat_names, "dep_var": dep_var}


def store_metadata(metadata, filepath):
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4)
    return


def load_metadata(filepath):
    metadata = {}
    with open(filepath) as json_file:
        metadata = json.load(json_file)
    return metadata


def analyse_df(df, options):

    _validate_options(options)
    metadata = _create_metadata(options)

    for proc in options["procs"]:
        if proc == "FillMissing":
            _analyse_fill_missing(df, options, metadata)
        elif proc == "Normalize":
            _analyse_standardization(df, options, metadata)
        elif proc == "Categorize":
            _analyse_categorization(df, options, metadata)
        else:
            raise ValueError("Unsupported proc type in options " + proc)

    metadata["columns"][options["dep_var"]] = {"type": COL_TYPES.DEPVAR.value}

    return metadata


def prepare_df(df, metadata):
    columns = list(df.columns.values)
    for column in columns:

        if column not in metadata["columns"]:
            raise ValueError("Unsupported column in df " + column)

        info = metadata["columns"][column]
        if info["type"] == COL_TYPES.NORMALIZED.value:
            df.loc[:, column] = _standardize_vec(df[column], info["mean"], info["std"])

        if info["type"] == COL_TYPES.CATEGORIZED.value:
            df.loc[:, column] = _map_vec(df[column], info["values"])

    return df
