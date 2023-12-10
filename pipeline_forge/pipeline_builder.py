import logging
from abc import ABC
from typing import Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .common.pipelines import FeatureEngineeringPipeline, ResamplingPipeline

log = logging.getLogger(__name__)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_pipeline):
        self.feature_pipeline = feature_pipeline

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        self.feature_pipeline.fit(X, y)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        return self.feature_pipeline.transform(X)


class FeaturePipelineBuilder(FeatureEngineeringPipeline, ResamplingPipeline, ABC):

    def __init__(self, target_col: str, random_seed: Optional[int] = 1995):
        self.target_col = target_col
        super().__init__(target_col=target_col)

    def build_pipeline(self) -> Pipeline:
        """
        Build the preconfigured Pipeline based on the added steps.

        Returns:
            Pipeline: Constructed Pipeline instance.
        """
        self.build_feature_engineering_steps()

        self.build_sampling_steps()

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
