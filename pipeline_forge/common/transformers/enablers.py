import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from pipeline_forge.utils import reduce_mem_usage


class DataframeMemoryReducer(BaseEstimator, TransformerMixin):
    """
    Transformer that reduces the size of an input Pandas dataframe by adjusting numerical features to smaller formats.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return reduce_mem_usage(df=X)
