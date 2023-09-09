import pandas as pd
from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin


class ProphetFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Transformer that adds features generated by Facebook Prophet to an input Pandas DataFrame.
    This transformer assumes there is both a 'Date' and 'Time' column.
    """

    def __init__(self, y_col='close'):
        self.y_col = y_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        # Combine 'Date' and 'Time' columns into a single 'Datetime' column
        X['Datetime'] = pd.to_datetime(X['Date'] + ' ' + X['Time'])

        # Select the relevant columns for Prophet
        df_prophet = X[['Datetime', self.y_col]]

        # Rename columns to match Prophet's expected format
        df_prophet = df_prophet.rename(columns={'Datetime': 'ds', self.y_col: 'y'})

        # Initialize and fit Prophet model with desired seasonality components
        model = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=True)
        model.fit(df_prophet)

        # Generate future dates for predictions (periods=0 for current dates)
        future = model.make_future_dataframe(periods=0)

        # Make predictions
        forecast = model.predict(future)

        # Rename columns for consistency
        forecast = forecast.rename(columns={'ds': 'Datetime'})

        # Drop unnecessary columns
        columns_to_drop = [
            'yhat'
        ]
        forecast.drop(columns=columns_to_drop, inplace=True)

        # Rename columns with a prefix (e.g., 'FBP_')
        forecast = forecast.rename(
            columns={col: f'FBP_{col.upper()}' for col in forecast.columns if col != 'Datetime'}
        )

        # Merge the forecasted features with the original DataFrame using a left join
        X = X.merge(forecast, on='Datetime', how='left')

        # Drop the created datetime column
        X.drop(columns=['Datetime'], inplace=True)

        return X


class DateTimeEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that encodes date and time columns into separate components and adds them as features.
    THis transformer assumes there is a 'Date' column.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # Extract date components
        X['YEAR'] = X['Date'].dt.year
        X['MONTH'] = X['Date'].dt.month
        X['DAY'] = X['Date'].dt.day
        X['DAY_OF_WEEK'] = X['Date'].dt.dayofweek

        # Extract time components
        X['HOUR'] = X['Time'].dt.hour
        X['MINUTE'] = X['Time'].dt.minute

        return X