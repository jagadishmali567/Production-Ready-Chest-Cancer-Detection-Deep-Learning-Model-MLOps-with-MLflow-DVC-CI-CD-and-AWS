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

