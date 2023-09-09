from abc import ABC
from typing import Any, Dict, List, Tuple

import mlflow
from sklearn.pipeline import Pipeline

from pipeline_forge.common.transformers import (
    CategoricalNonOrdinalTransformer, CategoricalNormTransformer,
    ColumnDropTransformer, CorrelationFeatureDrop, DataframeMemoryReducer,
    DateTimeEncoder, DecisionTreesFeatureSelector, FeatureScalerTransformer,
    FeatureSelector, FillNullsTransformer, ProphetFeatureGenerator,
    RandomForestFeatureSelector, SimpleImputerTransformer, WoETransformer)


class FeatureEngineeringPipeline(ABC):

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

    # Correlation feature drop transformer
    # Correlation threshold used to drop features
    CORRELATION_DROP_TH: float = None

    # Feature selection using a decision tree
    DT_IMPORTANCE_TH: float = None

    # Feature selection using a random forest
    RF_IMPORTANCE_TH: float = None

    # Add Date column encoder
    ENCODE_DATE: bool = False

    # Create prophet generated features
    USE_PROPHET: bool = False

    # Reduce memory transformer
    REDUCE_DF_MEM: bool = False

    def __init__(self):
        super().__init__()
        self.engineering_steps: List[Tuple[str], Any] = []

    def add_engineering_step(self, step_name: str, step_instance):
        """
        Add a step to the feature engineering pipeline.

        Args:
            step_name (str): Name of the pipeline step.
            step_instance: Instance of the transformer to be added as a step.

        Returns:
            FeaturePipelineBuilder: Updated builder instance.
        """
        self.engineering_steps.append((step_name, step_instance))
        return self

    def _build_feature_engineering_pipeline(self) -> Pipeline:
        """
        Build the preconfigured Pipeline based on the added steps.

        Returns:
            Pipeline: Constructed Pipeline instance.
        """

        if self.UNIQUE_FIELDS:
            self.add_engineering_step("drop_unique_fields", ColumnDropTransformer(drop_cols=self.UNIQUE_FIELDS))

        if self.SELECT_COLS:
            self.add_engineering_step("select_columns", FeatureSelector(select_cols=self.SELECT_COLS))

        if self.FILL_NA_DICT:
            self.add_engineering_step("fill_na", FillNullsTransformer(fill_na_dict=self.FILL_NA_DICT))

        if self.MEAN_IMPUTATION_NA_LIST:
            self.add_engineering_step("fill_na_mean", SimpleImputerTransformer(columns=self.MEAN_IMPUTATION_NA_LIST))

        if self.CATEGORICAL_COLS_WOE and self.TARGET_COL:
            self.add_engineering_step(
                "woe_categorical_imputer", WoETransformer(
                    categorical_cols=self.CATEGORICAL_COLS_WOE,
                    target_col=self.TARGET_COL))

        if self.REDUCE_DF_MEM:
            self.add_engineering_step("reduce_mem", DataframeMemoryReducer())

        if self.CAT_NORM_CAT_FEATURES and self.CAT_NORM_NUM_FEATURES and self.CAT_NORM_ALIASES:
            self.add_engineering_step(
                "cat_norm", CategoricalNormTransformer(
                    categorical_variables=self.CAT_NORM_CAT_FEATURES,
                    numerical_variables=self.CAT_NORM_NUM_FEATURES,
                    categorical_alias=self.CAT_NORM_ALIASES))

        if self.CAT_ONE_HOT_COLS:
            self.add_engineering_step(
                "one_hot_encode", CategoricalNonOrdinalTransformer(
                    non_ordinal_categorical_cols=self.CAT_ONE_HOT_COLS))

        if self.ENCODE_DATE:
            self.add_engineering_step(
                "date_encoder", DateTimeEncoder()
            )

        if self.USE_PROPHET:
            self.add_engineering_step(
                "prophet", ProphetFeatureGenerator()
            )

        if self.DROP_COLS:
            self.add_engineering_step("drop_columns", ColumnDropTransformer(drop_cols=self.DROP_COLS))

        # Start by logging no scaler and then overwrite with the correct technique if used
        if mlflow.active_run():
            mlflow.set_tag("scaler", None)

        if self.SCALER_TYPE:
            self.add_engineering_step("scaler", FeatureScalerTransformer(scaler_type=self.SCALER_TYPE))

            if mlflow.active_run():
                mlflow.set_tag("scaler", self.SCALER_TYPE)

        # Start by logging no correlation feature drop and then overwrite with the correct technique if used
        if mlflow.active_run():
            mlflow.set_tag("feature_correlation_drop", None)

        if self.CORRELATION_DROP_TH:
            self.add_engineering_step(
                "feature_correlation_drop", CorrelationFeatureDrop(
                    correlation_drop_th=self.CORRELATION_DROP_TH,
                    random_seed=self.RANDOM_SEED))

            if mlflow.active_run():
                mlflow.set_tag("feature_correlation_drop", self.CORRELATION_DROP_TH)

        # Start by logging no tree-based feature selection and then overwrite with the correct technique if used
        if mlflow.active_run():
            mlflow.set_tag("tree_based_fs", None)
        if self.DT_IMPORTANCE_TH and not self.SELECT_COLS:
            self.add_engineering_step(
                "decision_tree_feature_selection", DecisionTreesFeatureSelector(
                    dt_importance_th=self.DT_IMPORTANCE_TH,
                    random_seed=self.RANDOM_SEED))
            if mlflow.active_run():
                mlflow.set_tag("tree_based_fs", f"decision_tree: {self.DT_IMPORTANCE_TH}")

        if self.RF_IMPORTANCE_TH and not self.SELECT_COLS:
            self.add_engineering_step(
                "random_forest_feature_selection", RandomForestFeatureSelector(
                    rf_importance_th=self.RF_IMPORTANCE_TH,
                    random_seed=self.RANDOM_SEED))
            if mlflow.active_run():
                mlflow.set_tag("tree_based_fs", f"random_forest: {self.RF_IMPORTANCE_TH}")
