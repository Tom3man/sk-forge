from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
from sklearn.pipeline import Pipeline

from pipeline_forge.common.transformers import (
    CategoricalNonOrdinalTransformer, CategoricalNormTransformer,
    ColumnDropTransformer, CorrelationFeatureDrop, DataframeMemoryReducer,
    DateTimeEncoder, DecisionTreesFeatureSelector, FeatureScalerTransformer,
    FeatureSelector, FillNullsTransformer, ProphetFeatureGenerator,
    RandomForestFeatureSelector, SimpleImputerTransformer, WoETransformer)


class FeatureEngineeringPipeline(ABC):
    """
    A pipeline for feature engineering in machine learning tasks.

    This class provides methods to build a feature engineering pipeline for preprocessing
    and feature selection tasks. It allows you to add various data transformation steps
    to preprocess your data before training machine learning models.

    Args:
        target_col (str): The name of the target column.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 1995.
    """

    def __init__(self, target_col: str, random_seed: Optional[int] = 1995):
        """
        Initialize the FeatureEngineeringPipeline.

        Args:
            target_col (str): The name of the target column.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 1995.
        """
        super().__init__()
        self.target_col = target_col
        self.random_seed = random_seed
        self.engineering_steps: List[Tuple[str], Any] = []

    def drop_columns(self, drop_cols: List[str]) -> 'FeatureEngineeringPipeline':
        """
        Add a step to drop specified columns.

        Args:
            drop_cols (List[str]): List of column names to be dropped.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(("drop_columns", ColumnDropTransformer(drop_cols=drop_cols)))
        return self

    def select_columns(self, select_cols: List[str]) -> 'FeatureEngineeringPipeline':
        """
        Add a step to select specified columns.

        Args:
            select_cols (List[str]): List of column names to be selected.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(("select_columns", FeatureSelector(select_cols=select_cols)))
        return self

    def fill_na(self, fill_na_dict: Dict[str, Union[int, str, float]]) -> 'FeatureEngineeringPipeline':
        """
        Add a step to fill missing values in specified columns.

        Args:
            fill_na_dict (Dict[str, Union[int, str, float]]): A dictionary where keys are column names
                and values are fill values.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("fill_na", FillNullsTransformer(fill_na_dict=fill_na_dict))
        )
        return self

    def mean_imputation_na_list(self, fill_na_mean: List[str], strategy: str) -> 'FeatureEngineeringPipeline':
        """
        Add a step to perform mean imputation for specified columns.

        Args:
            fill_na_mean (List[str]): List of column names for mean imputation.
            strategy (str): Imputation strategy, one of 'mean', 'median', 'most_frequent', or 'constant'.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("fill_na_mean", SimpleImputerTransformer(
                columns=fill_na_mean,
                imputer_strategy=strategy
            ))
        )
        return self

    def woe_categorical_imputer(self, categorical_cols: List[str]) -> 'FeatureEngineeringPipeline':
        """
        Add a step to perform Weight of Evidence (WoE) transformation for specified categorical columns.

        Args:
            categorical_cols (List[str]): List of column names for WoE transformation.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("woe_categorical_imputer", WoETransformer(
                categorical_cols=categorical_cols,
                target_col=self.target_col))
        )
        return self

    def cat_norm_cat_features(
            self, categorical_variables: List[str],
            numerical_variables: List[int],
            categorical_alias: str,
    ) -> 'FeatureEngineeringPipeline':
        """
        Add a step to normalize specified categorical features.

        Args:
            categorical_variables (List[str]): List of categorical variable names.
            numerical_variables (List[int]): List of numerical variable indices.
            categorical_alias (str): Alias for the categorical features.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("cat_norm_cat_features", CategoricalNormTransformer(
                categorical_variables=categorical_variables,
                numerical_variables=numerical_variables,
                categorical_alias=categorical_alias))
        )
        return self

    def one_hot_encode(self, non_ordinal_categorical_cols: List[str]) -> 'FeatureEngineeringPipeline':
        """
        Add a step to perform one-hot encoding for specified non-ordinal categorical columns.

        Args:
            non_ordinal_categorical_cols (List[str]): List of non-ordinal categorical column names.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("one_hot_encode", CategoricalNonOrdinalTransformer(
                non_ordinal_categorical_cols=non_ordinal_categorical_cols))
        )
        return self

    def encode_date_col(self) -> 'FeatureEngineeringPipeline':
        """
        Add a step to encode date columns.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("date_encoder", DateTimeEncoder())
        )
        return self

    def add_prophet_features(self) -> 'FeatureEngineeringPipeline':
        """
        Add a step to generate features using Prophet.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("prophet", ProphetFeatureGenerator())
        )
        return self

    def reduce_mem_load(self) -> 'FeatureEngineeringPipeline':
        """
        Add a step to reduce memory usage of the DataFrame.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("reduce_mem", DataframeMemoryReducer())
        )
        return self

    def scaler(self, scaler_type: str) -> 'FeatureEngineeringPipeline':
        """
        Add a step to scale features using the specified scaler.

        Args:
            scaler_type (str): The type of scaler to use, one of 'standard', 'min_max', 'robust', or 'max_absolute'.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("scaler", FeatureScalerTransformer(scaler_type=scaler_type))
        )
        if mlflow.active_run():
            mlflow.set_tag("scaler", scaler_type)
        return self

    def drop_correlated_features(self, drop_thresh: float) -> 'FeatureEngineeringPipeline':
        """
        Add a step to drop correlated features based on the specified threshold.

        Args:
            drop_thresh (float): The correlation threshold used to drop features.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("feature_correlation_drop", CorrelationFeatureDrop(correlation_drop_th=drop_thresh))
        )
        if mlflow.active_run():
            mlflow.set_tag("correlated_feature_thresh", drop_thresh)
        return self

    def decision_tree_feat_select(self, importance_thresh: float) -> 'FeatureEngineeringPipeline':
        """
        Add a step to perform feature selection using decision trees.

        Args:
            importance_thresh (float): The importance threshold for feature selection.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("dt_importance_th", DecisionTreesFeatureSelector(
                importance_th=importance_thresh,
                random_seed=self.random_seed,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("dt_importance_th", importance_thresh)
        return self

    def random_forest_feat_select(self, importance_thresh: float) -> 'FeatureEngineeringPipeline':
        """
        Add a step to perform feature selection using random forests.

        Args:
            importance_thresh (float): The importance threshold for feature selection.

        Returns:
            FeatureEngineeringPipeline: Updated pipeline instance.
        """
        self.engineering_steps.append(
            ("rf_importance_th", RandomForestFeatureSelector(
                importance_th=importance_thresh,
                random_seed=self.random_seed,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("rf_importance_th", importance_thresh)
        return self

    def build_feature_engineering_steps(self) -> Pipeline:
        """
        Build the preconfigured Pipeline based on the added steps.

        Returns:
            Pipeline: Constructed Pipeline instance.
        """
        return Pipeline(self.engineering_steps)
