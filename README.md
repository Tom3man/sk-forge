# ml-pipeline-forge

Your go-to toolkit for crafting robust and flexible data pre-processing pipelines in Python. Whether you're working on machine learning projects, data analysis tasks, or data-driven applications, this repository empowers you to streamline the data preparation phase efficiently.

## Customisation

Using this pipeline builder is simple! All you have to do is build a training pipeline (and also a scoring pipeline if desired) by building a class that inherits from the FeaturePipelineBuilder class, overwritting the paramaters where appropiate.

The forge also incorporates MLFlow so will log certain aspects and stages of the pipeline where applicable.