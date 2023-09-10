import itertools
import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

log = logging.getLogger(__name__)


class SimpleImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Custom SimpleImputer adapted from sklearn to be able to select the columns
        to impute null values.

    Args:
        columns (List[str]): list of columns with null values to impute
        strategy: way to impute the missing values. Refer to SimpleImputer for more details
    """
    def __init__(self, columns: List[str], imputer_strategy: str):
        self.columns = columns
        self.imputer_strategy = imputer_strategy

        self.imputer_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.imputer_strategy not in ["mean", "median", "most_frequent", "constant"]:
            raise ValueError(f"Simple imputer strategy {self.imputer_strategy} not recognised, must be one of {', '.join(self.imputer_strategies)}")

        self.imputer = SimpleImputer(strategy=self.imputer_strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        log.info(f"Detected NaNs: {X[X.columns[X.isna().any()].tolist()].isna().sum()}")
        X_ = X.copy()
        X_[self.columns] = self.imputer.transform(X_[self.columns])
        log.info(f"After simple imputation: {X_[X_.columns[X_.isna().any()].tolist()].isna().sum()}")
        return X_


class FillNullsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that replace null values in a pandas dataframe with new values according to fill_na_dict.

    Args:
        fill_na_dict: dictionary where keys are columns and values are used to replace NaNs in that same column
    """
    def __init__(self, fill_na_dict: Dict[str, Any]):
        self.fill_na_dict = fill_na_dict

    def fit(self, X, y=None):
        if not all([col in X for col in self.fill_na_dict.keys()]):
            raise ValueError("Cols are not in the df")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        log.info(f"Detected NaNs: {X[X.columns[X.isna().any()].tolist()].isna().sum()}")
        log.info(f"Filling empty values with dict {self.fill_na_dict}")
        X_ = X.fillna(self.fill_na_dict)
        log.info(f"After simple imputation: {X_[X_.columns[X_.isna().any()].tolist()].isna().sum()}")

        return X_


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that implements Weights of Evidence method to convert categorical variables to numbers.
    https://towardsdatascience.com/churn-analysis-information-value-and-weight-of-evidence-6a35db8b9ec5
    """

    def __init__(self, categorical_cols: List[str],
                 target_col: str = "target"):
        """
        Args:
            categorical_cols: list of strings representing columns that should be treated as categorical
            and therefore will be transformed.
        """
        self.cat_cols = categorical_cols
        self.target_col = target_col
        self.woe_dfs = {cat_col: None for cat_col in self.cat_cols}

    def _check_inputs(self, df: pd.DataFrame):
        if not all([col in df.columns for col in self.cat_cols]):
            raise ValueError("Cat cols are not in df")

    def _group_by_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Args:
            df: pandas dataframe
            feature: name of categorical feature to group by

        Returns:
            Df with number of events/non events per each feature category
        """
        gr_df = df \
            .groupby(feature) \
            .agg({self.target_col: ['count', 'sum']}) \
            .reset_index()
        gr_df.columns = [feature, 'count', 'event']
        gr_df['non_event'] = gr_df['count'] - gr_df['event']
        return gr_df

    @staticmethod
    def _perc_share(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        return df[group_name] / df[group_name].sum()

    def _calculate_perc_share(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Calculates the percentage of events and non events for each category level of feature

        Args:
            df: pandas dataframe
            feature: name of categorical feature to calculate WoE for

        Returns:
            Pandas df
        """
        gr_df = self._group_by_feature(df=df,
                                       feature=feature)
        gr_df['perc_event'] = self._perc_share(gr_df, 'event')
        gr_df['perc_non_event'] = self._perc_share(gr_df, 'non_event')
        gr_df['perc_diff'] = gr_df['perc_event'] - gr_df['perc_non_event']
        return gr_df

    def _calculate_woe_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Calculate the WoE values for each category level of feature. If there are no events / non events in a
        category level, the WoE value will be equal to -1/1 accordingly.

        Args:
            df: pandas dataframe
            feature: name of categorical feature to calculate WoE for

        Returns:
            Pandas df with WoE values
        """
        gr_df = self._calculate_perc_share(df, feature)

        gr_df['woe'] = np.log(gr_df['perc_event'] / gr_df['perc_non_event'])

        if (gr_df['perc_event'] == 0).sum() > 0:
            log.warning(f"There are 0 events in {feature} - woe will be assigned to -1")
            gr_df['woe'] = gr_df['woe'].replace([-np.inf], np.nan).fillna(-1)

        if (gr_df['perc_non_event'] == 0).sum() > 0:
            log.warning(f"There are 0 non-events in {feature} - woe will be assigned to 1")
            gr_df['woe'] = gr_df['woe'].replace([np.inf], np.nan).fillna(1)
        return gr_df

    def fit_woe(self, df: pd.DataFrame, target: Union[np.ndarray, pd.Series]):
        """
        Receives a pandas df with categorical features and an array representing the ground truth for each
        sample. It calculates the WoE for each categorical feature given in self.cat_cols, and stores
        the result in a dictionary.

        Args:
            df: pandas dataframe containing the features to be transformed
            target: array or pandas Series containing the ground truth
        """
        self._check_inputs(df=df)
        if isinstance(target, np.ndarray):
            target = pd.Series(target)

        copy_df = df.copy()
        copy_df = pd.concat([copy_df, target.rename(self.target_col)], axis=1)

        for cat_col in self.cat_cols:
            woe_cat_df = self._calculate_woe_feature(copy_df, feature=cat_col)
            reduced_df = woe_cat_df[[cat_col, "woe"]].copy()
            reduced_df.columns = [cat_col, f"WOE_{cat_col}"]
            self.woe_dfs[cat_col] = reduced_df

    def transform_woe_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a pandas dataframe with categorical variables and applies the WoE transformation
        calculated in the fit step.

        Args:
            df: pandas dataframe

        Returns:
            pd.DataFrame: dataframe with the transformed features
        """
        # Done twice to be compliant with sklearn fit/transform
        self._check_inputs(df=df)
        df_ = df.copy()

        for cat_col, woe_df in self.woe_dfs.items():
            df_ = df_.merge(woe_df, on=cat_col, how="left")
        return df_

    def calculate_woe(self, df: pd.DataFrame, target: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
        """
        Runs fit and transform methods.

        Args:
            df: pandas dataframe containing the features to be transformed
            target:  array or pandas Series containing the ground truth

        Returns:
            pd.DataFrame: transformed dataframe
        """
        self.fit_woe(df=df, target=target)
        return self.transform_woe_variables(df=df)

    def fit(self, X, y):
        log.info("Calculating WoEs")
        self.fit_woe(df=X, target=y)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        log.info("Encoding using WoEs")
        X_woe = self.transform_woe_variables(df=X)
        log.info(f"Shape: {X_woe.shape}")
        return X_woe


class CategoricalNormTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn compatible transformer that normalises a list of numerical variables with respect a list
    of categorical variables. For a pair of numerical and categorical variables,
    the transformer takes the mean and std from each category, and it then transforms the numerical variable,
    by taking the mean and dividing by the std in each category.

    Args:
        categorical_variables: list of categorical variables to be used as reference
        numerical_variables: list of numerical variables to be transformed
        categorical_alias: list of aliases to append each transformed numerical column
    """

    def __init__(self, categorical_variables: List[str],
                 numerical_variables: List[str] = None,
                 categorical_alias: List[str] = None):
        self.categorical_variables = categorical_variables
        self.numerical_variables = numerical_variables
        self.categorical_alias = categorical_alias if categorical_alias is not None else categorical_variables
        self.cat_var_dict = {cat_var: {num_var: None for num_var in self.numerical_variables}
                             for cat_var in self.categorical_variables}

    def _init_cat_var(self, df: pd.DataFrame):
        """
        Calculates the mean and std for every combination of category and numerical variable, and stores it
        in an intermediate dictionary.

        Args:
            df: pd.DataFrame
        """
        for cat_var, num_var in itertools.product(self.categorical_variables, self.numerical_variables):
            gr_df = df.groupby(cat_var, as_index=False)\
                      .agg(feat_mean=(num_var, 'mean'),
                           feat_std=(num_var, 'std'))
            if len(gr_df[gr_df.feat_std == 0]) != 0:
                log.warning(f"Skipping {cat_var} and {num_var} combination - Zero variance in a group")
                continue
            self.cat_var_dict[cat_var][num_var] = gr_df

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("This transformer only works with Pandas dfs")

        if not all([cat_var in X.columns for cat_var in self.categorical_variables]):
            raise ValueError("Provided variables are not in the input df")

        if self.numerical_variables is not None and not all([num_var in X.columns for num_var in self.numerical_variables]):
            raise ValueError("Provided variables are not in the input df")

        self.numerical_variables = self.numerical_variables if self.numerical_variables is not None else X.columns

        if not all([is_numeric_dtype(X[num_var]) for num_var in self.numerical_variables]):
            raise ValueError("numerical_variables must only include numerical features")

        self._init_cat_var(df=X)
        return self


class CategoricalNonOrdinalTransformer(BaseEstimator, TransformerMixin):
    """
    Creates dummy columns for non-ordinal data.

    Args:
        non_ordinal_categorical_cols List[str]:
            List of column names to be one-hot encoded as categorical variables.

    Returns:
        pd.DataFrame: The DataFrame with feature engineering transformations applied.

    Raises:
        ValueError: If there is an error while encoding ordinal categorical columns.
    """

    def __init__(
        self,
        non_ordinal_categorical_cols: List[str],
    ):
        self.non_ordinal_categorical_cols = non_ordinal_categorical_cols
        self.fitted_categorical_cols = []
        self.dummy_column_names = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all([cat_var in X.columns for cat_var in self.non_ordinal_categorical_cols]):
            raise ValueError("Provided variables are not in the input df")

        self.fitted_categorical_cols = self.non_ordinal_categorical_cols

        # Create dummy columns for training data and store their names
        X_encoded = pd.get_dummies(
            X, columns=self.fitted_categorical_cols, prefix='CATEGORY'
        )
        self.dummy_column_names = X_encoded.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # Ensure the same categorical columns as during fitting are present
        if not all([cat_var in X.columns for cat_var in self.fitted_categorical_cols]):
            raise ValueError("Provided variables are not in the input df")

        # Create dummy columns for test data using stored column names
        X_encoded = pd.get_dummies(
            X, columns=self.fitted_categorical_cols, prefix='CATEGORY'
        )

        # Add missing dummy columns with values of zero
        missing_cols = set(self.dummy_column_names) - set(X_encoded.columns)
        for col in missing_cols:
            X_encoded[col] = 0

        # Reorder columns to match the order in which they were created during fitting
        X_encoded = X_encoded[self.dummy_column_names]

        return X_encoded
