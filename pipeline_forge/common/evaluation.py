import logging
from abc import ABC
from typing import Any, Callable, Dict, List, Union

import mlflow
import numpy as np
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, mean_squared_log_error,
                             median_absolute_error, r2_score)

log = logging.getLogger(__name__)


class BuildModelEvaluationMetrics(ABC):
    """
    Abstract base class for building model evaluation metrics.
    """

    def __init__(
            self,
            model,
            X: Union[np.ndarray, List[Union[int, float]]],
            y: Union[np.ndarray, List[Union[int, float]]],
    ):
        """
        Initialize the BuildModelEvaluationMetrics.

        Args:
            model (keras.Model): Trained keras network model.
            X (np.ndarray): Test input data.
            y (np.ndarray): Test target data.
        """
        self.model = model
        self.X = X
        self.y = y

        self.eval_dict = dict()

    def _calculate_huber_loss(self, delta: float):
        """
        Calculate Huber loss.

        Args:
            delta (float): Huber loss parameter.
        """
        errors = self.y - self.y_pred
        quadratic = np.minimum(np.abs(errors), delta)
        linear = np.abs(errors) - quadratic
        huber_loss = 0.5 * quadratic ** 2 + delta * linear

        if huber_loss is not None and isinstance(huber_loss, (int, float)):
            self.eval_dict['h_loss'] = huber_loss
            mlflow.log_metric('h_loss', huber_loss)

    def _calculate_quantile_loss(self, quantile: float):
        """
        Calculate quantile loss.

        Args:
            quantile (float): Quantile value.
        """
        errors = self.y - self.y_pred
        mask_underestimation = errors < 0
        quantile_loss = (quantile - 1) * np.sum(errors[mask_underestimation]) + quantile * np.sum(errors[~mask_underestimation])
        quantile_loss_05 = quantile_loss / len(self.y)

        if quantile_loss_05 is not None and isinstance(quantile_loss_05, (int, float)):
            self.eval_dict['quantile_loss_05'] = quantile_loss_05
            mlflow.log_metric('quantile_loss_05', quantile_loss_05)

    def _calculate_mean_squared_log_error(self):
        """
        Calulate Mean Squared Log Error (MSLE) metric.
        """
        try:
            msle = mean_squared_log_error(self.y, self.y_pred)
            if msle is not None and isinstance(msle, (int, float)):
                self.eval_dict['msle'] = msle
                mlflow.log_metric('msle', msle)
        except ValueError:
            log.debug("Mean squared log error was unable to be extracted.")

    def _calculate_mean_squared_error(self):
        """
        Calulate Mean Squared Error (MSE) metric.
        """
        try:
            mse = mean_squared_error(self.y, self.y_pred)
            if mse is not None and isinstance(mse, (int, float)):
                self.eval_dict['mse'] = mse
                mlflow.log_metric('mse', mse)
        except ValueError:
            log.debug("Mean squared error was unable to be extracted.")

    def _calculate_mean_absolute_error(self):
        """
        Calulate Mean Absolute Error (MAE) metric.
        """
        try:
            mae = mean_absolute_error(self.y, self.y_pred)
            if mae is not None and isinstance(mae, (int, float)):
                self.eval_dict['mae'] = mae
                mlflow.log_metric('mae', mae)
        except ValueError:
            log.debug("Mean absolute error was unable to be extracted.")

    def _calculate_r2(self):
        """
        Calulate Coefficient of Determination (R-squared) metric.
        """
        try:
            r2 = r2_score(self.y, self.y_pred)
            if r2 is not None and isinstance(r2, (int, float)):
                self.eval_dict['r2'] = r2
                mlflow.log_metric('r2', r2)
        except ValueError:
            log.debug("R2 was unable to be extracted.")

    def _calculate_root_mean_squared_error(self):
        """
        Calulate Root Mean Squared Error (RMSE) metric.
        """
        try:
            mse = mean_squared_error(self.y, self.y_pred)
            rmse = np.sqrt(mse)
            if rmse is not None and isinstance(rmse, (int, float)):
                self.eval_dict['rmse'] = rmse
                mlflow.log_metric('rmse', rmse)
        except ValueError:
            log.debug("Root mean squared error was unable to be extracted.")

    def _calculate_median_absolute_error(self):
        """
        Calulate Median Absolute Error (MedAE) metric.
        """
        try:
            medae = median_absolute_error(self.y, self.y_pred)
            if medae is not None and isinstance(medae, (int, float)):
                self.eval_dict['medae'] = medae
                mlflow.log_metric('medae', medae)
        except ValueError:
            log.debug("Median absolute error was unable to be extracted.")

    def _calculate_explained_variance(self):
        """
        Calulate Explained Variance metric.
        """
        try:
            explained_var = explained_variance_score(self.y, self.y_pred)
            if explained_var is not None and isinstance(explained_var, (int, float)):
                self.eval_dict['explained_var'] = explained_var
                mlflow.log_metric('explained_var', explained_var)
        except ValueError:
            log.debug("Explained Variance absolute error was unable to be extracted.")

    def _calculate_mean_absolute_percentage_error(self):
        """
        Calculate Mean Absolute Percentage Error (MAPE) metric.
        """
        try:
            mape = np.mean(np.abs((self.y - self.y_pred) / np.maximum(self.y, 1))) * 100
            if mape is not None and isinstance(mape, (int, float)):
                self.eval_dict['mape'] = mape
                mlflow.log_metric('mape', mape)
        except ValueError:
            log.debug("Mean absolute percentage error was unable to be extracted.")

    def _calculate_relative_absolute_error(self):
        """
        Calulate Relative Absolute Error (RAE) metric.
        """
        try:
            rae = np.sum(np.abs(self.y - self.y_pred)) / np.sum(np.abs(self.y - np.mean(self.y)))
            if rae is not None and isinstance(rae, (int, float)):
                self.eval_dict['rae'] = rae
                mlflow.log_metric('rae', rae)
        except ValueError:
            log.debug("Relative absolute error was unable to be extracted.")

    def _calculate_symmetric_mean_absolute_percentage_error(self):
        """
        Calulate Symmetric Mean Absolute Percentage Error (SMAPE) metric.
        """
        try:
            smape = 100 * np.mean(2 * np.abs(self.y_pred - self.y) / (np.abs(self.y_pred) + np.abs(self.y)))
            if smape is not None and isinstance(smape, (int, float)):
                self.eval_dict['smape'] = smape
                mlflow.log_metric('smape', smape)
        except ValueError:
            log.debug("Symmetric mean absolute percentage error was unable to be extracted.")

    def _calculate_coefficient_of_determination(self):
        """
        Calulate Coefficient of Determination (COD) metric.
        """
        try:
            cod = 1 - np.sum(np.square(self.y - self.y_pred)) / np.sum(np.square(self.y - np.mean(self.y)))
            if cod is not None and isinstance(cod, (int, float)):
                self.eval_dict['cod'] = cod
                mlflow.log_metric('cod', cod)
        except ValueError:
            log.debug("Coefficient of Determination was unable to be extracted.")


class ModelEvaluation(BuildModelEvaluationMetrics):
    """
    A utility class for evaluating the performance of a Keras model using various evaluation metrics.

    This class provides a convenient interface to calculate and collect a range of evaluation metrics for regression
    models. It supports metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Coefficient of
    Determination (R-squared), and many others. Users can easily configure and calculate multiple metrics in one go
    using the provided methods.

    The calculated metrics can be accessed through the `evaluate()` method after adding the desired metrics using
    the dedicated methods. This class serves as a useful tool for quantifying the performance of machine learning
    regression models.

    Args:
        model (keras.models.Model): The trained Keras model to be evaluated.
        X (np.ndarray or List[Union[int, float]]): Test feature data.
        y (np.ndarray or List[Union[int, float]]): Ground truth test labels.

    Example:
            evaluate = ModelEvaluation(
                model=model,
                X=X_transformed,
                y=y,
            )
            evaluate.add_huber_loss()
            evaluate.add_quantile_loss()
            evaluate.add_mean_squared_log_error()
            evaluate.add_mean_squared_error()
            evaluate.add_mean_absolute_error()
            evaluate.add_root_mean_squared_error()
            evaluate.add_r2()
            evaluate.add_median_absolute_error()
            evaluate.add_explained_variance()
            evaluate.add_mean_absolute_percentage_error()
            evaluate.add_relative_absolute_error()
            evaluate.add_symmetric_mean_absolute_percentage_error()
            evaluate.add_coefficient_of_determination()

            evaluate.evaluate()
    """

    def __init__(
            self,
            model,
            X: Union[np.ndarray, List[Union[int, float]]],
            y: Union[np.ndarray, List[Union[int, float]]],
    ):
        """
        Initialize a ModelEvaluation instance.

        Args:
            model (keras.models.Model): The trained Keras model to be evaluated.
            X (np.ndarray or list): Test feature data.
            y (np.ndarray or list): Ground truth test labels.
        """
        super().__init__(model=model, X=X, y=y)

        self.model = model
        self.X = X
        self.y = y

        self.eval_metrics: List[str] = []

        # Dictionary mapping metric names to corresponding metric methods.
        self.metric_methods: Dict[str, Callable] = {
            'h_loss': self._calculate_huber_loss,
            'quantile_loss_05': self._calculate_quantile_loss,
            'msle': self._calculate_mean_squared_log_error,
            'mse': self._calculate_mean_squared_error,
            'mae': self._calculate_mean_absolute_error,
            'rmse': self._calculate_root_mean_squared_error,
            'r2': self._calculate_r2,
            'medae': self._calculate_median_absolute_error,
            'explained_var': self._calculate_explained_variance,
            'mape': self._calculate_mean_absolute_percentage_error,
            'rae': self._calculate_relative_absolute_error,
            'smape': self._calculate_symmetric_mean_absolute_percentage_error,
            'cod': self._calculate_coefficient_of_determination,
        }

    def add_huber_loss(self, delta: float = 0.1) -> 'ModelEvaluation':
        """
        Add Huber Loss metric.

        Huber Loss is a robust loss function that combines the benefits of Mean Squared Error (MSE) and Mean Absolute
        Error (MAE). It is less sensitive to outliers compared to MSE and provides a smoother loss curve for optimization.

        Choosing a good default value for delta can depend on the specific problem you're working on and the characteristics of your data.
        However, a common approach is to set delta to a value that is proportional to the scale of your target variable.
        For example, if your target variable is in the range of 0 to 100, a value like 1.0 or 2.0 for delta might be reasonable.

        Args:
            delta (float): Huber loss parameter.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('h_loss', delta))
        return self

    def add_quantile_loss(self, quantile: float = 0.75) -> 'ModelEvaluation':
        """
        Add Quantile Loss metric.

        Quantile Loss measures the weighted sum of underestimation and overestimation errors for a given quantile level.
        It is used to evaluate the accuracy of prediction intervals and provides insight into the model's performance
        across different quantiles.

        Args:
            quantile (float): Quantile value.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('quantile_loss_05', quantile))
        return self

    def add_mean_squared_log_error(self) -> 'ModelEvaluation':
        """
        Add Mean Squared Log Error (MSLE) metric.

        MSLE measures the mean squared logarithmic difference between the actual and predicted values. It is suitable
        for data with large magnitude differences and provides a way to penalize large prediction errors.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('msle', ))
        return self

    def add_mean_squared_error(self) -> 'ModelEvaluation':
        """
        Add Mean Squared Error (MSE) metric.

        MSE measures the average of the squared differences between the actual and predicted values. It provides insight
        into the overall model accuracy and the magnitude of errors.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('mse', ))
        return self

    def add_mean_absolute_error(self) -> 'ModelEvaluation':
        """
        Add Mean Absolute Error (MAE) metric.

        MAE measures the average of the absolute differences between the actual and predicted values. It provides insight
        into the average magnitude of errors and is less sensitive to outliers compared to MSE.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('mae', ))
        return self

    def add_root_mean_squared_error(self) -> 'ModelEvaluation':
        """
        Add Root Mean Squared Error (RMSE) metric.

        RMSE measures the square root of the average of the squared differences between the actual and predicted values.
        It is a variant of Mean Squared Error (MSE) and provides insight into the magnitude of errors while being in the
        same unit as the target variable.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('rmse', ))
        return self

    def add_r2(self) -> 'ModelEvaluation':
        """
        Add Coefficient of Determination (R-squared) metric.

        R-squared measures the proportion of the variance in the dependent variable that is explained by the independent
        variables. It ranges from 0 to 1, where higher values indicate a better fit of the model to the data.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('r2', ))
        return self

    def add_median_absolute_error(self) -> 'ModelEvaluation':
        """
        Add Median Absolute Error (MedAE) metric.

        MedAE measures the median of the absolute differences between the actual and predicted values. It provides insight
        into the central tendency of errors and is less sensitive to outliers compared to MAE.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('medae', ))
        return self

    def add_explained_variance(self) -> 'ModelEvaluation':
        """
        Add Explained Variance metric.

        Explained Variance measures the proportion of the variance in the dependent variable that is explained by the model.
        It provides insight into how well the model's predictions match the variability in the target variable.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('explained_var', ))
        return self

    def add_mean_absolute_percentage_error(self) -> 'ModelEvaluation':
        """
        Add Mean Absolute Percentage Error (MAPE) metric.

        MAPE measures the percentage difference between the actual and predicted values, normalized by actual values.
        It provides insight into the relative error of the model's predictions and is commonly used in forecasting tasks.
        The lower the MAPE, the better the model's accuracy.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('mape', ))
        return self

    def add_relative_absolute_error(self) -> 'ModelEvaluation':
        """
        Add Relative Absolute Error (RAE) metric.

        RAE measures the ratio of the sum of absolute errors to the sum of absolute errors obtained by predicting
        the mean of the target variable. It provides information about how well the model's errors compare to a
        naive baseline. Lower RAE values indicate better model performance.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('rae', ))
        return self

    def add_symmetric_mean_absolute_percentage_error(self) -> 'ModelEvaluation':
        """
        Add Symmetric Mean Absolute Percentage Error (SMAPE) metric.

        SMAPE is similar to MAPE but ensures symmetric treatment of overestimation and underestimation.
        It provides insight into the relative error of the model's predictions and is commonly used in forecasting tasks.
        The lower the SMAPE, the better the model's accuracy.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('smape', ))
        return self

    def add_coefficient_of_determination(self) -> 'ModelEvaluation':
        """
        Add Coefficient of Determination (COD) metric.

        COD, also known as R-squared, measures the proportion of the variance in the dependent variable that is predictable
        from the independent variables. It ranges from 0 to 1, where higher values indicate a better fit of the model
        to the data. However, it may not be suitable for models with non-linear relationships.

        Returns:
            ModelEvaluation: Current ModelEvaluation instance.
        """
        self.eval_metrics.append(('cod', ))
        return self

    def evaulate_all(self):
        self.add_huber_loss()
        self.add_quantile_loss()
        self.add_mean_squared_log_error()
        self.add_mean_squared_error()
        self.add_mean_absolute_error()
        self.add_root_mean_squared_error()
        self.add_r2()
        self.add_median_absolute_error()
        self.add_explained_variance()
        self.add_mean_absolute_percentage_error()
        self.add_relative_absolute_error()
        self.add_symmetric_mean_absolute_percentage_error()
        self.add_coefficient_of_determination()

        return self.evaluate()

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model's performance using the specified metrics on the test set.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation metrics and their corresponding values.
        """

        # Evaluate the model on the test set
        self.X = self.X.astype(np.float32)
        self.y_pred = self.model.predict(self.X)
        self.y_pred = self.y_pred.reshape(-1)

        for metric, *args in self.eval_metrics:
            method = self.metric_methods.get(metric)
            if method:
                method(*args)

        return self.eval_dict
