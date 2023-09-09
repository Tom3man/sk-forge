import logging
from abc import ABC
from typing import Any, Dict, List, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .pipelines import FeatureEngineeringPipeline, ResamplingPipeline

log = logging.getLogger(__name__)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_pipeline):
        self.feature_pipeline = feature_pipeline

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        self.feature_pipeline.fit(X, y)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        return self.feature_pipeline.transform(X)


class FeaturePipelineBuilder(ResamplingPipeline, FeatureEngineeringPipeline, ABC):

    # Target column
    TARGET_COL: str = None

    RANDOM_SEED: int = 1995

    # List of unique identifier columns for the drop columns transformer
    UNIQUE_FIELDS: List[str] = None

    # List of columns to drop for the drop columns transformer
    DROP_COLS: List[str] = None

    # List of columns used for direct feature selector
    SELECT_COLS: List[str] = None

    # The type of scaler to use for the scaler transformer
    # Can be one of 'standard', 'min_max', 'robust' or 'max_absolute'
    SCALER_TYPE: str = None

    # Dictionary where the keys are columns and values are fills
    # Used to fill NA values, used by the fill null transformer
    FILL_NA_DICT: Dict[str, Any] = None

    # WOE transformer
    CATEGORICAL_COLS_WOE: List[str] = None

    # List of columns to impute, used by the mean imputer transformer
    MEAN_IMPUTATION_NA_LIST: List[str] = None
    # Stratgy used by the mean imputer transformer
    # Can be either "mean", "median", "most_frequent" or "constant"
    MEAN_IMPUTATION_STRATEGY: str = None

    # Used by the categorical normaliser transformer
    # List of categorical variables to be used as reference
    CAT_NORM_CAT_FEATURES: List[str] = None
    # List of numerical variables to be transformed
    CAT_NORM_NUM_FEATURES: List[str] = None
    # List of aliases to append each transformed numerical column
    CAT_NORM_ALIASES: List[str] = None

    # One-hot encode transformer
    # Lost of columns to create dummy columns with
    CAT_ONE_HOT_COLS: List[str] = None

    # Add Date column encoder
    ENCODE_DATE: bool = False

    # Create prophet generated features
    USE_PROPHET: bool = False

    # Correlation feature drop transformer
    # Correlation threshold used to drop features
    CORRELATION_DROP_TH: float = None

    # Feature selection using a decision tree
    DT_IMPORTANCE_TH: float = None

    # Feature selection using a random forest
    RF_IMPORTANCE_TH: float = None

    # Over sampling
    # Use random over sampling
    RANDOM_OVERSAMPLE: float = None

    # SMOTE parameters
    USE_SMOTE: bool = False
    SMOTE_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'k_neighbors': 5,
        'sampling_strategy': 0.5,
        'kind': 'danger',
    }

    # ADASYN parameters
    USE_ADASYN: bool = False
    ADASYN_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'n_neighbors': 5,
        'sampling_strategy': 'auto',
    }

    # Borderline-SMOTE parameters
    USE_BORDERLINE_SMOTE: bool = False
    BORDERLINE_SMOTE_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'sampling_strategy': 0.5,
        'kind': 'borderline-1',
        'k_neighbors': 5,
    }

    # Under sampling
    # Random Undersampling parameters
    USE_RANDOM_UNDERSAMPLING: bool = False
    RANDOM_UNDERSAMPLING_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'sampling_strategy': 0.5,
    }

    # Cluster Centroids parameters
    USE_CLUSTER_CENTROIDS: bool = False
    CLUSTER_CENTROIDS_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'sampling_strategy': 'auto',
    }

    # Tomek Links parameters
    USE_TOMEK_LINKS: bool = False
    TOMEK_LINKS_PARAMETERS: Dict[str, Union[str, int, float]] = {}

    # ENN parameters
    USE_ENN: bool = False
    ENN_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'sampling_strategy': 'auto',
    }

    # NearMiss parameters
    USE_NEAR_MISS: bool = False
    NEAR_MISS_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'sampling_strategy': 'auto',
        'version': 1,
    }

    # Reduce memory transformer
    REDUCE_DF_MEM: bool = False

    def __init__(self):
        super().__init__()

    def build_pipeline(self) -> Pipeline:
        """
        Build the preconfigured Pipeline based on the added steps.

        Returns:
            Pipeline: Constructed Pipeline instance.
        """
        self._build_feature_engineering_pipeline()

        self._assemble_undersampling_pipeline()
        self._assemble_oversampling_pipeline()

        if not len(self.engineering_steps) > 0:
            raise ValueError("Not enough arguments have been passed to create a Pipeline")

        # Create the feature engineering pipeline
        engineer_pipeline = Pipeline(self.engineering_steps)

        # Create the imbalanced pipeline if needed
        if len(self.imbalanced_steps) > 0:
            imbalanced_pipeline = ImbalancedPipeline(self.imbalanced_steps)
            # Create a custom transformer that manages both pipelines

            self.full_pipeline = ImbalancedPipeline([
                ('feature', FeatureEngineeringTransformer(engineer_pipeline)),
                ('sampling', imbalanced_pipeline)
            ])
        else:
            self.full_pipeline = engineer_pipeline

        return self.full_pipeline

    @property
    def pipeline(self) -> Pipeline:
        """
        Function that returns a preconfigured Pipeline. This method can be used in different steps of a CV process,
        where a new pipeline is needed at each step.

        Returns:
            Pipeline
        """
        return self.full_pipeline

    def fit_transform_features(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
        """
        Fit the pipeline on the training data and transform it.

        Parameters:
            X (array-like): Input features.
            y (array-like): Target labels.

        Returns:
            np.ndarray: Transformed data.
        """
        starting_df_cols = len(X.columns)

        if not len(self.imbalanced_steps) > 0:
            fit_transformed = self.full_pipeline.fit_transform(X, y)
        else:
            fit_transformed = self.full_pipeline.named_steps['feature'].fit_transform(X, y)

        percentage_decrease = ((starting_df_cols - len(fit_transformed.columns)) / starting_df_cols)

        if mlflow.active_run():
            mlflow.set_tag("feature_decrease_perc", f"{percentage_decrease * 100:.2f}%")

        return fit_transformed

    def fit_features(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
        """
        Fit the pipeline on the training data.

        Parameters:
            X (array-like): Input features.
            y (array-like): Target labels.
        """
        if not len(self.imbalanced_steps) > 0:
            return self.full_pipeline.fit(X, y)
        return self.full_pipeline.named_steps['feature'].fit(X, y)

    def transform_features(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Transforms the fitted pipeline on the testing/validation data.

        Parameters:
            X (array-like): Input features.
        """
        if not len(self.imbalanced_steps) > 0:
            return self.full_pipeline.transform(X)
        return self.full_pipeline.named_steps['feature'].transform(X)

    def fit_resample_features(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit and resample the pipeline on the training data.

        Parameters:
            X (array-like): Input features.
            y (array-like): Target labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Resampled data and corresponding labels.
        """
        if mlflow.active_run():
            mlflow.set_tag("imbalance_target_change_perc", None)

        if not len(self.imbalanced_steps) > 0:
            log.info("Not enough arguments have been passed to create a sampling Pipeline")
            return X, y

        starting_target_true = len([i for i in y if i == 1])
        X_resampled, y_resampled = self.full_pipeline.named_steps['sampling'].fit_resample(X, y)
        resampled_target_true = len([i for i in y_resampled if i == 1])
        target_perc_change = ((resampled_target_true - starting_target_true) / starting_target_true) * 100

        if mlflow.active_run():
            mlflow.set_tag("imbalance_target_change_perc", target_perc_change)

        return X_resampled, y_resampled
