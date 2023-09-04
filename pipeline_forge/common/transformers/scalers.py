import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, RobustScaler,
                                   StandardScaler)

log = logging.getLogger(__name__)


class FeatureScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Scale the features of the input DataFrame X while respecting the month column.

    Args:
        scaler_type (str): The type of scaler to use.
        month_col (str): The name of the column representing months. Defaults to "SUMMARY_MTH".

    Returns:
        pd.DataFrame: The scaled X DataFrame containing scaled values and the month column.

    Raises:
        ValueError: If an unsupported scaler is requested.
    """

    def __init__(
        self,
        scaler_type: str,
        month_col: str = "SUMMARY_MTH",
    ):
        self.month_col = month_col
        self.scaler_type = scaler_type

        self.scalers = {
            "standard": StandardScaler(),
            "min_max": MinMaxScaler(),
            "robust": RobustScaler(),
            "max_absolute": MaxAbsScaler(),
        }

    def fit(self, X: pd.DataFrame, y=None):
        if self.scaler_type not in self.scalers.keys():
            raise ValueError(f"Scaler requested '{self.scaler_type}' not found in available scalers. Supported scalers: {', '.join(self.scalers.keys())}.")
        log.info(f"Scaling X with {self.scaler_type} scaler type.")
        self.scaler = self.scalers[self.scaler_type]
        self.scaler.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
        )

        return X_scaled
