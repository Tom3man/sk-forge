from abc import ABC
from typing import Any, List, Optional, Tuple, Union

import mlflow
from imblearn.over_sampling import (ADASYN, SMOTE, BorderlineSMOTE,
                                    RandomOverSampler)
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import (ClusterCentroids, EditedNearestNeighbours,
                                     NearMiss, RandomUnderSampler, TomekLinks)


class ResamplingPipeline(ABC):
    """
    A pipeline for handling imbalanced datasets by applying various resampling techniques.

    This class provides methods to build a pipeline for resampling imbalanced datasets
    using techniques such as oversampling and undersampling. It helps address class imbalance
    issues in machine learning tasks.

    Attributes:
        imbalanced_steps (List[Tuple[str, Any]]): A list of tuples representing the steps
            in the resampling pipeline.
    """

    def __init__(self):
        super().__init__()
        self.imbalanced_steps: List[Tuple[str], Any] = []

    def random_oversampling(self, sampling_strategy: Union[float, str]) -> 'ResamplingPipeline':
        """
        Apply random oversampling to balance the class distribution.

        Random oversampling involves randomly duplicating instances from the minority class
        until it's balanced with the majority class. This technique can lead to overfitting
        since it replicates existing data.

        Args:
            sampling_strategy (Union[float, str]): The strategy for oversampling. Can be a float
                specifying the desired ratio or 'auto' to balance classes.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("random_oversampling", RandomOverSampler(
                sampling_strategy=sampling_strategy,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("random_oversampling_strategy", sampling_strategy)
        return self

    def smote_oversampling(
            self, sampling_strategy: Union[float, str],
            k_neighbors: Optional[int] = 5
    ) -> 'ResamplingPipeline':
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) oversampling.

        SMOTE creates synthetic samples by interpolating between existing samples.
        It selects a sample from the minority class and finds its k nearest neighbors.
        Then, it creates new samples along the line segments connecting the sample and its neighbors.

        Args:
            sampling_strategy (Union[float, str]): The strategy for oversampling. Can be a float
                specifying the desired ratio or 'auto' to balance classes.
            k_neighbors (Optional[int]): The number of nearest neighbors to consider.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("smote_oversampling", SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("smote_oversampling_strategy", sampling_strategy)
            mlflow.set_tag("smote_oversampling_kn", k_neighbors)
        return self

    def adasyn_oversampling(
            self,
            sampling_strategy: Union[float, str],
            k_neighbors: Optional[int] = 5
    ) -> 'ResamplingPipeline':
        """
        Apply ADASYN (Adaptive Synthetic Sampling) oversampling.

        ADASYN focuses on generating more samples for the minority class instances that are difficult to classify.
        It generates samples proportionally to the density of the class distribution.

        Args:
            sampling_strategy (Union[float, str]): The strategy for oversampling. Can be a float
                specifying the desired ratio or 'auto' to balance classes.
            k_neighbors (Optional[int]): The number of nearest neighbors to consider.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("adasyn_oversampling", ADASYN(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("adasyn_oversampling", sampling_strategy)
            mlflow.set_tag("adasyn_oversampling", k_neighbors)
        return self

    def borderline_smote_oversampling(
            self,
            sampling_strategy: Union[float, str],
            k_neighbors: Optional[int] = 5,
            kind: str = 'borderline-1',
    ) -> 'ResamplingPipeline':
        """
        Apply Borderline-SMOTE oversampling.

        Similar to SMOTE, but it focuses on generating synthetic samples near the decision boundary,
        where the classes are difficult to separate.

        Args:
            sampling_strategy (Union[float, str]): The strategy for oversampling. Can be a float
                specifying the desired ratio or 'auto' to balance classes.
            k_neighbors (Optional[int]): The number of nearest neighbors to consider.
            kind (str): The type of Borderline-SMOTE to use.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("borderline_smote_oversampling", BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                kind=kind,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("borderline_smote_oversampling", sampling_strategy)
            mlflow.set_tag("borderline_smote_oversampling", k_neighbors)
        return self

    def random_undersampling(self, sampling_strategy: Union[float, str]) -> 'ResamplingPipeline':
        """
        Apply random undersampling to balance the class distribution.

        Random undersampling involves randomly removing instances from the majority class
        until it's balanced with the minority class. Like random oversampling, it can result in loss of information.

        Args:
            sampling_strategy (Union[float, str]): The strategy for undersampling. Can be a float
                specifying the desired ratio or 'auto' to balance classes.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("random_undersampling", RandomUnderSampler(
                sampling_strategy=sampling_strategy,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("random_oversampling_strategy", sampling_strategy)
        return self

    def cluster_centoids_undersampling(self, sampling_strategy: Union[float, str]) -> 'ResamplingPipeline':
        """
        Apply Cluster Centroids undersampling.

        Cluster Centroids undersampling identifies clusters of majority class instances
        and then removes the majority class instances that are closest to the centroid of each cluster.

        Args:
            sampling_strategy (Union[float, str]): The strategy for undersampling. Can be a float
                specifying the desired ratio or 'auto' to balance classes.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("cluster_centoids_undersampling", ClusterCentroids(
                sampling_strategy=sampling_strategy,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("cluster_centoids_undersampling", sampling_strategy)
        return self

    def tomek_links_undersampling(self) -> 'ResamplingPipeline':
        """
        Apply Tomek Links undersampling.

        A Tomek link is a pair of instances from different classes that are closest to each other.
        Removing the majority class instance from such pairs can help in improving class separation.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("tomek_links_undersampling", TomekLinks())
        )
        return self

    def enn_undersampling(self, sampling_strategy: Union[float, str]) -> 'ResamplingPipeline':
        """
        Apply Edited Nearest Neighbors (ENN) undersampling.

        ENN removes instances whose class label differs from the majority class label
        of its k nearest neighbors.

        Args:
            sampling_strategy (Union[float, str]): The strategy for undersampling. Can be a float
                specifying the desired ratio or 'auto' to balance classes.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("enn_undersampling", EditedNearestNeighbours(
                sampling_strategy=sampling_strategy,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("enn_undersampling", sampling_strategy)
        return self

    def near_miss_undersampling(self, sampling_strategy: Union[float, str], version: int = 1) -> 'ResamplingPipeline':
        """
        Apply NearMiss undersampling.

        NearMiss selects examples from the majority class that are near the minority class instances
        based on some distance metric.

        Args:
            sampling_strategy (Union[float, str]): The strategy for undersampling. Can be a float
                specifying the desired ratio or 'auto' to balance classes.
            version (int): The version of the NearMiss algorithm to use.

        Returns:
            ResamplingPipeline: Updated pipeline instance.
        """
        self.imbalanced_steps.append(
            ("near_miss_undersampling", NearMiss(
                sampling_strategy=sampling_strategy,
                version=version,
            ))
        )
        if mlflow.active_run():
            mlflow.set_tag("near_miss_undersampling", sampling_strategy)
        return self

    def build_sampling_steps(self) -> ImbalancedPipeline:
        """
        Build the preconfigured Pipeline based on the added steps.

        Returns:
            Pipeline: Constructed Pipeline instance.
        """
        return ImbalancedPipeline(self.imbalanced_steps)
