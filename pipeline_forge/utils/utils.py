import logging
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def dict_params_to_int(dict_params: Dict, int_params: List) -> Dict:
    """
    It cast each parameter in int_params to integer in dict_params. All of int_params should be
    keys in dict_params.

    Args:
        dict_params: dictionary where keys are parameters names and values are parameters values
        int_params: list of parameters in dict_params to be casted to integer

    Returns:
        dict: modified dictionary
    """
    params_copy = dict_params.copy()
    for int_param in int_params:
        params_copy[int_param] = int(round(params_copy[int_param]))
    return params_copy


def save_pickle(v, path: str):
    """
    Saves v in a pickle file under path
    Args:
        v: python variable to store
        path: str
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(v, file)


def read_pickle(path: str):
    """
    Reads and stores the content of a pickle located in path

    Args:
        path: str representing a file path

    Returns:
        pickle result
    """
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces df memory by modifying numerical columns format based on max and min values. It assumes that the
    dataframe has no empty values.

    Args:
        df: pandas dataframe

    Returns:
        pd.Dataframe
    """
    def get_optimized_dtype(col):
        max_val = col.max()
        min_val = col.min()

        if col.dtype.kind == 'i':
            return np.iinfo(col.dtype).min, np.iinfo(col.dtype).max

        if np.isfinite(max_val) and np.isfinite(min_val):
            return np.float32

    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    log.info(f"Memory usage of properties dataframe is : {start_mem_usg} MB")

    for col in df.columns:
        if df[col].dtype != object:
            optimized_dtype = get_optimized_dtype(df[col])

            if optimized_dtype:
                df[col] = df[col].astype(optimized_dtype)

    log.info("MEMORY USAGE AFTER COMPLETION:")
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    log.info(f"Memory usage is: {mem_usg} MB")
    log.info(f"This is {100 * mem_usg / start_mem_usg:.2f}% of the initial size")
    return df
