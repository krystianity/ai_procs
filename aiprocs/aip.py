import json
import pandas as pd
import numpy as np
from enum import Enum

__version__ = "0.2.0"


class MODEL_TYPES(Enum):
    PYTORCH = "PYTORCH"
    TENSORFLOW = "TENSORFLOW"


class COL_TYPES(Enum):
    NORMALIZED = "NORMALIZED"
    CATEGORIZED = "CATEGORIZED"
    DEPVAR = "DEPVAR"


class FILL_TYPES(Enum):
    MEDIAN = "MEDIAN"
    CONST = "CONST"


def _create_metadata(options):
    return {"version": __version__, "options": options, "columns": {}}


def _standardize_vec(vec, mean, std):
    return vec.sub(mean).div(std)


def _map_vec(vec, hashmap):
    return vec.map(hashmap)


def _analyse_fill_missing(df, options, metadata):
    columns = list(df.columns.values)
    ignore_list = options["cat_names"].copy()
    ignore_list.append(options["dep_var"])
    for column in columns:
        if column not in ignore_list:
            strategy = options["fill_strategy"]
            if strategy == FILL_TYPES.MEDIAN:
                median = df[column].median()
                metadata["columns"][column] = {"fill_type": strategy, "median": median}
            elif strategy == FILL_TYPES.CONST:
                metadata["columns"][column] = {
                    "fill_type": strategy,
                    "fill_val": options["filler"],
                }


def _analyse_standardization(df, options, metadata):
    columns = list(df.columns.values)
    ignore_list = options["cat_names"].copy()
    ignore_list.append(options["dep_var"])
    for column in columns:
        if column not in ignore_list:
            mean = df[column].mean()
            std = df[column].std()
            if column not in metadata["columns"]:
                metadata["columns"][column] = {
                    "type": COL_TYPES.NORMALIZED.value,
                    "mean": mean,
                    "std": std,
                }
            else:
                metadata["columns"][column] = {
                    "type": COL_TYPES.NORMALIZED.value,
                    "mean": mean,
                    "std": std,
                    **metadata["columns"][column],
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


def _apply_fill_missing(vec, column_options):
    strategy = column_options["fill_type"]
    if strategy == FILL_TYPES.MEDIAN.value:
        median = column_options["median"]
        vec.fillna(median)
    elif strategy == FILL_TYPES.CONST.value:
        filler = column_options["filler"]
        vec.fillna(filler)
    return vec


def _validate_options(options):
    if "procs" not in options:
        raise ValueError("Missing procs key in options dict")
    if "cat_names" not in options:
        raise ValueError("Missing cat_names key in options")
    if "dep_var" not in options:
        raise ValueError("Missing dep_var key in options dict")


def split_df(df, valid_frac=0.2, test_frac=0.0):
    """
    Splits a dataset into 3 parts, after randomizing its rows.
    Returns the rest as training data.
    """
    df_len = len(df)
    random_idx = np.random.permutation(df_len)
    df = df.iloc[random_idx]
    valid_cut = int(valid_frac * df_len)
    test_cut = valid_cut + int(test_frac * df_len)
    valid_df = df.iloc[:valid_cut]
    test_df = df.iloc[valid_cut:test_cut]
    train_df = df.iloc[test_cut:]
    return train_df, valid_df, test_df


def create_options(procs, cat_names, dep_var, model_type):
    """
    Creates options for aiprocs, that can be passed to the analyse methods.
        procs => ["FillMissing", "Normalize", "Categorify"]
        cat_names => ["cat1", "cat2"]
        dep_var => predict column (output)
        model_type => choose from MODEL_TYPES
    """
    return {
        "procs": procs,
        "cat_names": cat_names,
        "dep_var": dep_var,
        "fill_strategy": FILL_TYPES.MEDIAN.value,
        "filler": 0,
        "model_type": model_type,
    }


def store_metadata(metadata, filepath):
    """
        Stores metadata as JSON file in given path.
    """
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4)
    return


def load_metadata(filepath):
    """
        Load metadata from JSON file from given path.
    """
    metadata = {}
    with open(filepath) as json_file:
        metadata = json.load(json_file)
    return metadata


def analyse_df(df, options):
    """
        Analyses a dataframe, creating a metadata object
        based on the passed options. Metadata objects can be
        used to apply dataset preparations across multiple platforms.
    """
    _validate_options(options)
    metadata = _create_metadata(options)

    for proc in options["procs"]:
        if proc == "FillMissing":
            _analyse_fill_missing(df, options, metadata)
        elif proc == "Normalize":
            _analyse_standardization(df, options, metadata)
        elif proc == "Categorify":
            _analyse_categorization(df, options, metadata)
        else:
            raise ValueError("Unsupported proc type in options " + proc)

    metadata["columns"][options["dep_var"]] = {"type": COL_TYPES.DEPVAR.value}

    return metadata


def prepare_df(df, metadata):
    """
        Alters a dataframe depending on the passed metadata object.
    """
    columns = list(df.columns.values)
    for column in columns:

        if column not in metadata["columns"]:
            raise ValueError("Unsupported column in df " + column)
        info = metadata["columns"][column]

        if "fill_type" in info:
            df.loc[:, column] = _apply_fill_missing(df[column], info)

        if info["type"] == COL_TYPES.NORMALIZED.value:
            df.loc[:, column] = _standardize_vec(df[column], info["mean"], info["std"])

        if info["type"] == COL_TYPES.CATEGORIZED.value:
            df.loc[:, column] = _map_vec(df[column], info["values"])

    return df


def prepare_dfl(df_list, metadata):
    """
        Alter multiple dataframes depending on the passed metadata object.
    """
    return [prepare_df(df, metadata) for df in df_list]
