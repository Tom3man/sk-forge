from abc import ABC
from typing import Any, Dict, List, Tuple, Union
import mlflow
from imblearn.over_sampling import (ADASYN, SMOTE, BorderlineSMOTE,
                                    RandomOverSampler)
from imblearn.under_sampling import (ClusterCentroids, EditedNearestNeighbours,
                                     NearMiss, RandomUnderSampler, TomekLinks)


class ResamplingPipeline(ABC):

    # Over sampling
    # Use random over sampling
    USE_RANDOM_OVERSAMPLING: float = False
    RANDOM_OVERSAMPLING_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'sampling_strategy': 0.5,
    }

    # SMOTE parameters
    USE_SMOTE: bool = False
    SMOTE_PARAMETERS: Dict[str, Union[str, int, float]] = {
        'k_neighbors': 5,
        'sampling_strategy': 0.5,
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

    def __init__(self):
        super().__init__()
        self.imbalanced_steps: List[Tuple[str], Any] = []

    def add_imbalanced_step(self, step_name: str, step_instance):
        """
        Add a step to the imbalance resampling pipeline.

        Args:
            step_name (str): Name of the pipeline step.
            step_instance: Instance of the transformer to be added as a step.

        Returns:
            FeaturePipelineBuilder: Updated builder instance.
        """
        self.imbalanced_steps.append((step_name, step_instance))
        return self

    def _assemble_oversampling_pipeline(self):
        """
        Method used for different oversampling strategies. All types are specific for binary classification tasks.

        Random Oversampling:
            This involves randomly duplicating instances from the minority class until it's balanced with the majority class. While simple, it can lead to overfitting since it's essentially replicating existing data.

        SMOTE (Synthetic Minority Over-sampling Technique):
            SMOTE creates synthetic samples by interpolating between existing samples. It selects a sample from the minority class and finds its k nearest neighbors. Then, it creates new samples along the line segments connecting the sample and its neighbors.

        ADASYN (Adaptive Synthetic Sampling):
            ADASYN focuses on generating more samples for the minority class instances that are difficult to classify. It generates samples proportionally to the density of the class distribution.

        Borderline-SMOTE:
            Similar to SMOTE, but it focuses on generating synthetic samples near the decision boundary, where the classes are difficult to separate.
        """

        if sum([self.USE_RANDOM_OVERSAMPLING, self.USE_SMOTE, self.USE_ADASYN, self.USE_BORDERLINE_SMOTE]) > 1:
            raise AssertionError("We recommend using only one oversampling technique.")

        # Start by logging no oversampling and then overwrite with the correct technique if used
        if mlflow.active_run():
            mlflow.set_tag("oversampling", None)

        # Build over sampling pipeline
        if self.USE_RANDOM_OVERSAMPLING:
            self.add_imbalanced_step(
                "random_over_sampling", RandomOverSampler(
                    sampling_strategy=self.RANDOM_OVERSAMPLING_PARAMETERS
                )
            )
            if mlflow.active_run():
                mlflow.set_tag(
                    "oversampling", f"random_over_sampling - {self.RANDOM_OVERSAMPLING_PARAMETERS['sampling_strategy']}")

        elif self.USE_SMOTE and self.SMOTE_PARAMETERS:
            self.add_imbalanced_step(
                "smote_over_sampling", SMOTE(
                    **self.SMOTE_PARAMETERS
                )
            )
            if mlflow.active_run():
                mlflow.set_tag(
                    "oversampling", f"smote_over_sampling - {self.SMOTE_PARAMETERS['sampling_strategy']}")

        elif self.USE_ADASYN and self.ADASYN_PARAMETERS:
            self.add_imbalanced_step(
                "adasyn_over_sampling", ADASYN(
                    **self.ADASYN_PARAMETERS
                )
            )
            if mlflow.active_run():
                mlflow.set_tag(
                    "oversampling", f"adasyn_over_sampling - {self.ADASYN_PARAMETERS['sampling_strategy']}")

        elif self.USE_BORDERLINE_SMOTE and self.BORDERLINE_SMOTE_PARAMETERS:
            self.add_imbalanced_step(
                "borderline_smote_over_sampling", BorderlineSMOTE(
                    **self.BORDERLINE_SMOTE_PARAMETERS
                )
            )

            if mlflow.active_run():
                mlflow.set_tag(
                    "oversampling", f"borderline_smote_over_sampling - {self.BORDERLINE_SMOTE_PARAMETERS['sampling_strategy']}")

    def _assemble_undersampling_pipeline(self):
        """
        Method for different undersampling strategies. All types are specific for binary classification tasks.

        Random Undersampling:
            This involves randomly removing instances from the majority class until it's balanced with the minority class. Like random oversampling, it can result in loss of information.

        Cluster Centroids:
            Cluster Centroids undersampling identifies clusters of majority class instances and then removes the majority class instances that are closest to the centroid of each cluster.

        Tomek Links:
            A Tomek link is a pair of instances from different classes that are closest to each other. Removing the majority class instance from such pairs can help in improving class separation.

        ENN (Edited Nearest Neighbors):
            ENN removes instances whose class label differs from the majority class label of its k nearest neighbors.

        NearMiss:
            NearMiss selects examples from the majority class that are near the minority class instances based on some distance metric.
        """

        if sum([
            self.USE_RANDOM_UNDERSAMPLING, self.USE_CLUSTER_CENTROIDS,
            self.USE_TOMEK_LINKS, self.USE_ENN, self.USE_NEAR_MISS
        ]) > 1:
            raise AssertionError("We recommend using only one undersampling technique.")

        if mlflow.active_run():
            mlflow.set_tag("undersampling", None)

        # Build under sampling pipeline
        if self.USE_RANDOM_UNDERSAMPLING and self.RANDOM_UNDERSAMPLING_PARAMETERS:
            self.add_imbalanced_step(
                "random_undersampling", RandomUnderSampler(
                    **self.RANDOM_UNDERSAMPLING_PARAMETERS
                )
            )

            if mlflow.active_run():
                mlflow.set_tag(
                    "undersampling", f"random_undersampling - {self.RANDOM_UNDERSAMPLING_PARAMETERS['sampling_strategy']}")

        elif self.USE_CLUSTER_CENTROIDS and self.CLUSTER_CENTROIDS_PARAMETERS:
            self.add_imbalanced_step(
                "cluster_centroids", ClusterCentroids(
                    **self.CLUSTER_CENTROIDS_PARAMETERS
                )
            )

            if mlflow.active_run():
                mlflow.set_tag(
                    "undersampling", f"cluster_centroids - {self.CLUSTER_CENTROIDS_PARAMETERS['sampling_strategy']}")

        elif self.USE_TOMEK_LINKS and self.TOMEK_LINKS_PARAMETERS:
            self.add_imbalanced_step(
                "tomek_links", TomekLinks(
                    **self.TOMEK_LINKS_PARAMETERS
                )
            )

            if mlflow.active_run():
                mlflow.set_tag("undersampling", "tomek_links")

        elif self.USE_ENN and self.ENN_PARAMETERS:
            self.add_imbalanced_step(
                "enn", EditedNearestNeighbours(
                    **self.ENN_PARAMETERS
                )
            )

            if mlflow.active_run():
                mlflow.set_tag(
                    "undersampling", f"enn - {self.ENN_PARAMETERS['sampling_strategy']}")

        elif self.USE_NEAR_MISS and self.NEAR_MISS_PARAMETERS:
            self.add_imbalanced_step(
                "near_miss", NearMiss(
                    **self.NEAR_MISS_PARAMETERS
                )
            )

            if mlflow.active_run():
                mlflow.set_tag(
                    "undersampling", f"near_miss - {self.NEAR_MISS_PARAMETERS['sampling_strategy']}")
