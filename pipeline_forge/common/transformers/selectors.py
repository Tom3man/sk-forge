import logging
from typing import Dict, List, Tuple, Type, Set

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

log = logging.getLogger(__name__)


class DecisionTreesFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that selects features by fitting a decision tree or a random forest to a dataset,
    and selecting the features that represent a percentage of the total importance explained by the model.
    In other words, it chooses the top features that a tree or ensemble of trees most uses when fitting
    a model to the data. In order for this to work, the model should be quite overfitted.

    Args:
        importance_th (float):
            Threshold that represents the percentage of explained importance that wants to be captured with the features.
        random_seed (int):
            Random seed for reproducibility.
        model_type (str):
            Model type ("dt" for decision tree or "rf" for random forest).
    """

    def __init__(
            self,
            importance_th: float,
            random_seed: int = None,
            model_type: str = 'dt'
    ):
        self.importance_th = importance_th
        self.random_seed = random_seed

        if model_type not in self.tree_models.keys():
            raise ValueError(f"Only {list(self.tree_models.keys())} are supported")

        self.model_type = model_type
        self.train_loss = None
        self.selected_features = None
        self.selected_features_importance = None

    @property
    def tree_models(self) -> Dict[str, Type[BaseEstimator]]:
        """
        Dictionary of supported tree models.

        Returns:
            dict: A dictionary mapping model type to the corresponding tree model instance.
        """
        return {
            "dt": {
                "regressor": DecisionTreeRegressor(random_state=self.random_seed),
                "classifier": DecisionTreeClassifier(random_state=self.random_seed),
            },
            "rf": {
                "regressor": RandomForestRegressor(random_state=self.random_seed),
                "classifier": RandomForestClassifier(bootstrap=False, random_state=self.random_seed),
            },
        }

    @staticmethod
    def is_continuous(y: np.ndarray) -> bool:
        """
        Check if the target variable is continuous.

        Args:
            y (np.ndarray): The target variable.

        Returns:
            bool: True if the target variable is continuous, False otherwise.
        """
        unique_values = np.unique(y)
        return np.issubdtype(unique_values.dtype, np.number)

    def _compute_train_loss(self, X: pd.DataFrame, y):
        """
        Compute the train loss of the model.

        Args:
            X (pd.DataFrame): Feature data.
            y (array-like): Target labels.
        """
        if self.model_class == 'regressor':
            self.train_loss = mean_squared_error(y, self.model.predict(X))
        elif self.model_class == "classifier":
            self.train_loss = log_loss(y, self.model.predict_proba(X))
        log.info(f"Train loss: {self.train_loss}")

    def _select_based_on_importance(self) -> Tuple[List, List]:
        """
        Calculate the cumulative importance given by the model and select features based on a threshold.

        Returns:
            Tuple[List, List]: List of selected features and a list of importance values for those features.
        """
        if self.model_class == "regressor":
            if not hasattr(self.model, 'feature_importances_'):
                raise ValueError("The model does not have feature_importances_ attribute.")

            importance_df = pd.DataFrame({
                "features": self.feature_names_train,  # Replace with actual feature names or indices from training data
                "importance": self.model.feature_importances_
            })
        elif self.model_class == "classifier":
            importance_df = pd.DataFrame({
                "features": self.model.feature_names_in_,
                "importance": self.model.feature_importances_
            })

        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['cumulative_importance'] = importance_df.importance.cumsum()

        selected_features_df = importance_df[importance_df.cumulative_importance < self.importance_th]
        return selected_features_df.features.values, selected_features_df.importance.values

    def fit(self, X: pd.DataFrame, y):
        """
        Train an overfitted tree/tree ensemble and select features that sum up to a percentage of the importance.

        Args:
            X (pd.DataFrame): Feature data.
            y (array-like): Target labels.

        Returns:
            TreeBasedFeatureSelector: The fitted transformer instance.
        """

        if self.is_continuous(y=y):
            self.model_class = 'regressor'
        else:
            self.model_class = 'classifier'

        self.model = self.tree_models[self.model_type][self.model_class]

        X_num = X.select_dtypes("number")
        log.info(f"Overfitting a {self.model_class} type {self.model_type} for {X_num.shape[1]} numerical variables")

        self.model = self.model.fit(X_num, y)
        self.feature_names_train = X_num.columns.tolist()  # Store the feature names from training data
        self._compute_train_loss(X=X_num, y=y)
        self.selected_features, self.selected_features_importance = self._select_based_on_importance()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform the input DataFrame by selecting features based on importance.

        Args:
            X (pd.DataFrame): Input data.
            y: Not used.

        Returns:
            pd.DataFrame: Transformed DataFrame with selected features.
        """
        log.info(f"Selecting {len(self.selected_features)} features")
        X_sel = X[self.selected_features].copy()
        log.info(f"Shape: {X_sel.shape}")
        return X_sel


class RandomForestFeatureSelector(DecisionTreesFeatureSelector):
    """
    The Random Forest implementation of a tree based selector transformer.
    """
    def __init__(self, importance_th: float, random_seed: int = None):
        super().__init__(
            importance_th=importance_th,
            random_seed=random_seed,
            model_type='rf'
        )


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that selects select_cols features from a pandas dataframe.
    This transformer should be used in the train and score pipeline builds but can be bypassed with one of the feature selection transformers for the expermenetal train stages.

    Args:
        select_cols (List[str]): List of columns to select
    """

    def __init__(self, select_cols: List[str]):
        self.select_cols = select_cols

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.select_cols:
            if col not in X.columns:
                raise ValueError(f"{col} is not in X")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        log.info(f"Selecting {len(self.select_cols)} cols:  {self.select_cols}")

        X_sel = X[self.select_cols].copy()
        log.info(f"Shape: {X_sel.shape}")
        return X_sel


class CorrelationFeatureDrop(BaseEstimator, TransformerMixin):
    """
    Transformer that drops features from a pandas dataframe based on linear correlations.
    For each pair of features, if the linear correlation between them is higher than a threshold,
    one of the features will be randomly dropped.

    Args:
        correlation_drop_th (float):
            Correlation threshold used to drop features.
        random_seed (int, optional):
            Integer representing a random seed. Set this number for replicable results.
    """

    def __init__(self, correlation_drop_th: float, random_seed: int = None):
        self.correlation_drop_th = correlation_drop_th
        self.random_seed = random_seed
        self.correlated_cols_to_drop: Set[str] = set()
        self.correlation_pairs: List[Tuple[str, str]] = []

    @staticmethod
    def drop_cols_correlation_th(
            corr_df: pd.DataFrame,
            th: float,
            seed: int = None
    ) -> Tuple[Set[str], List[Tuple[str, str]]]:
        """
        Returns a set of columns to drop based on correlations given by corr_df.
        It evaluates each pair of features and randomly drops one column from the pair
        if the correlation is higher than a threshold.

        Args:
            corr_df (pd.DataFrame):
                DataFrame containing the correlations between features.
            th (float):
                Threshold used to decide whether to drop a column from a pair of correlated features.
            seed (int, optional):
                Integer representing a random seed.

        Returns:
            Set[str]: Set of columns to drop.
            List[Tuple[str, str]]: List of tuples indicating pairs of columns with a correlation higher than th.
        """
        indices = np.where(abs(corr_df) > th)
        indices = [(corr_df.index[x], corr_df.columns[y]) for x, y in zip(*indices) if x != y and x < y]
        np.random.seed(seed)
        drop_cols = set(map(lambda x: x[np.random.randint(0, 2)], indices))
        return drop_cols, indices

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by computing correlations and identifying columns to drop.

        Args:
            X (pd.DataFrame):
                Feature data.
            y:
                Not used.

        Returns:
            CorrelationFeatureDrop: The fitted transformer instance.
        """
        X_num = X.select_dtypes("number")
        log.info(f"Computing linear correlations for {X_num.shape[1]} numerical variables")
        corr_matrix = X_num.corr()
        self.correlated_cols_to_drop, self.correlation_pairs = self.drop_cols_correlation_th(
            corr_df=corr_matrix,
            th=self.correlation_drop_th,
            seed=self.random_seed
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform the input DataFrame by dropping correlated features.

        Args:
            X (pd.DataFrame):
                Input data.
            y:
                Not used.

        Returns:
            pd.DataFrame: Transformed DataFrame with correlated features dropped.
        """
        log.info(f"Dropping {len(self.correlated_cols_to_drop)} columns using th={self.correlation_drop_th}")
        X_drop = X.drop(self.correlated_cols_to_drop, axis=1)
        log.info(f"Shape: {X_drop.shape}")
        return X_drop


class ColumnDropTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that drops drop_cols from a pandas dataframe

    Args:
        drop_cols (List[str]]): List of column names to drop.

    Returns:
        pd.DataFrame: The DataFrame with feature engineering transformations applied.
    """

    def __init__(self, drop_cols: List[str]):
        self.drop_cols = drop_cols

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all([col in X.columns for col in self.drop_cols]):
            raise ValueError(f"Provided variables {', '.join(self.drop_cols)} are not in the input df")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        log.info(f"Dropping unique fields: {', '.join(self.drop_cols)}")

        # Drop unique identifiers if provided
        if self.drop_cols:
            X_drop = X.drop(columns=self.drop_cols)
            log.info(f"Shape: {X_drop.shape}")

        return X_drop
