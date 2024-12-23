# Production-Ready Chest Cancer Detection: Deep Learning Model, MLOps with MLflow, DVC, CI/CD, and AWS

### Project Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

### Git commands

```Bash
git add .
```

```Bash
git commit -m "Updated"
```

```Bash
git push origin main
```


### How to run?

```Bash
conda create -n ccancer python=3.8 -y
```

```Bash
source activate base
```

```Bash
conda activate ccancer
```

```Bash
pip install -r requirements.txt
```


### MLflow
- [Documentation](https://mlflow.org/docs/latest/index.html)

cmd
```Bash
mlflow ui
```


### Dagshub

Run this to export as env variables:
```Bash  
export MLFLOW_TRACKING_URI=https://dagshub.com/jagadishmali567/Production-Ready-Chest-Cancer-Detection-Deep-Learning-Model-MLOps-with-MLflow-DVC-CI-CD-and-AWS.mlflow
```

### Data Version Control (DVC)
**cmd**
```Bash
dvc init
```
```Bash
dvc repro
```
```Bash
dvc dag
```

## About MLflow and DVC (Data Version Control)

### MLflow:

**MLflow** is an open-source platform designed to manage the machine learning lifecycle. It offers a suite of tools to streamline the process of developing, managing, and deploying machine learning models. Here are some key components and benefits of MLflow:

1. **Tracking:** MLflow Tracking logs parameters, code versions, metrics, and artifacts, making it easier to compare different runs and experiments.

2. **Model Registry:** This component helps manage and track different versions of models, ensuring smooth transitions from development to production.

3. **Projects:** MLflow Projects package code into reusable and reproducible formats, facilitating collaboration and deployment.

4. **Models:** It supports various machine learning libraries and frameworks, enabling easy deployment of models to different environments.

5. **Model Serving:** MLflow provides REST endpoints for hosting models, making it easier to integrate them into applications.

### DVC (Data Version Control):

**DVC** is an open-source tool that helps manage data and machine learning models, similar to how Git manages code. It focuses on versioning data and ensuring reproducibility in ML projects. Here are some key roles and benefits of DVC:

1. **Data Versioning:** DVC tracks changes in datasets, ensuring that we can reproduce experiments and collaborate effectively.

2. **Pipeline Management:** It allows us to define and manage data processing pipelines, making it easier to automate and reproduce workflows.

3. **Remote Storage:** DVC supports remote storage solutions, enabling efficient storage and sharing of large datasets.

4. **Reproducibility:** By versioning both code and data, DVC ensures that experiments can be reproduced accurately.

5. **Collaboration:** DVC facilitates collaboration among team members by providing a unified interface for managing data and models.


### Example Scenario

Imagine we're working on a machine learning project to predict housing prices. Here's how we might use MLflow and DVC:

1. **Tracking Experiments with MLflow:**

   - We run multiple experiments with different hyperparameters to optimize our model.

   - MLflow logs the parameters, metrics (like RMSE and MAE), and artifacts (like model files) for each run.

   - We can easily compare the performance of different runs and select the best model.

2. **Managing Data with DVC:**

   - We version our dataset with DVC, ensuring we can reproduce our experiments later.

   - DVC tracks changes to the dataset and stores it in a remote storage like an S3 bucket.

   - When collaborating with team members, they can pull the exact version of the dataset we used for our experiments.

3. **Model Deployment with MLflow:**

   - Once we have the best model, we register it in the MLflow Model Registry.

   - MLflow provides a REST endpoint to serve our model, allowing us to integrate it into our application seamlessly.

4. **Reproducing Experiments with DVC and MLflow:**

   - Months later, we decide to revisit the project. With DVC, we retrieve the exact version of the dataset we used.

   - Using MLflow, we reproduce the experiments and deploy an updated version of the model with new data.

*By combining MLflow and DVC, we can effectively manage the entire machine learning lifecycle, from data versioning and experiment tracking to model deployment and collaboration.*


## Automating Amazon Web Services(AWS) CI/CD Deployments through GitHub Actions

### 1. Login to AWS Console.
### 2. Create a new IAM user with programmatic access.

**with specific access**

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


**Description: About the deployment**

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

**Policy:**

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess

