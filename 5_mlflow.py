# Databricks notebook source
# MAGIC %md ## Logging and tracking models using MLflow  
# MAGIC  - See the [MLflow documentation](https://www.mlflow.org/docs/latest/index.html)
# MAGIC  - There are two key components to MLflow: **Experiments**, which is like a sandbox area for recordning model training runs, and the **Registry**, which is a way to isolate and manage production models or models considered for production.  
# MAGIC  - We first log model artifacts and metrics to an Experiment. Then, there is an option to move one or more of the models to the Registry.
# MAGIC  - Models are loaded from the Registry for inference

# COMMAND ----------

from collections import OrderedDict
from typing import List


from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support


from pyspark.sql.functions import col
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, SparkTrials
from hyperopt.early_stop import no_progress_loss

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from helpers import get_current_user

# COMMAND ----------

# MAGIC %md #### Create MLflow experiment  
# MAGIC We will log our fitted model and performance metrics to this location. All aspects of MLflow can be controlled using the MLflow API. We will use the API to set up an Experiment for this project as well as an entry in the Registry. The documentation [here](https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html) and [here](https://www.mlflow.org/docs/latest/rest-api.html) is helpful for understanding the API.

# COMMAND ----------

def get_or_create_experiment(experiment_location: str) -> None:
 
  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)

  
current_user = get_current_user()  
experiment_location = f'/Shared/{current_user}_experiment'
get_or_create_experiment(experiment_location)

mlflow.set_experiment(experiment_location)

# COMMAND ----------

# MAGIC %md #### Create synthentic dataset  
# MAGIC We will train a classifier on this dataset

# COMMAND ----------

n_samples = 1000
n_features = 20

X, y = make_classification(n_samples=     n_samples, 
                           n_features=    n_features, 
                           n_informative= n_features, 
                           n_redundant=    0, 
                           n_classes=      2, 
                           flip_y=         0.4,
                           random_state=   np.random.randint(1,999))

# Create missing values to impute
observations = n_samples * n_features
missing_proportion = 0.1
missing_observations = int(missing_proportion * observations)

X.ravel()[np.random.choice(X.size, missing_observations, replace=False)] = None

# Create Pandas DataFrame of numeric features
numeric_feature_names = [f'numeric_feature_{str(n+1).zfill(2)}' for n in range(n_features)]
features_df = pd.DataFrame(X, columns=numeric_feature_names)
features_df['label'] = y

# Add a categorical column
categories = ['A', 'B', 'C']
cat_probs_1 = [0.7, 0.2, 0.1]
cat_probs_0 = [0.2, 0.6, 0.2]

features_df.loc[features_df['label'] == 1, 'categorical_feature_01'] = np.random.choice(categories, features_df.loc[features_df['label'] == 1].shape[0], p=cat_probs_1)
features_df.loc[features_df['label'] == 0, 'categorical_feature_01'] = np.random.choice(categories, features_df.loc[features_df['label'] == 0].shape[0], p=cat_probs_0)

col_ordering = ['label', 'categorical_feature_01'] + numeric_feature_names

features_df[col_ordering].head()

# COMMAND ----------

# MAGIC %md #### Scikit-learn pre-processing pipeline

# COMMAND ----------

def create_preprocessing_transform(categorical_features: List[str], numerical_features: List[str]) -> ColumnTransformer:
  
  categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy = 'constant', fill_value = 'missing')),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

  numerical_pipe_quantile = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

  preprocessing = ColumnTransformer(
        [
            ("cat", categorical_pipe, categorical_features),
            ("num", numerical_pipe_quantile, numerical_features)
        ],
        remainder='drop'
    )

  return preprocessing

# COMMAND ----------

# MAGIC %md #### Train the model  
# MAGIC  - split into train/test datasets
# MAGIC  - fit model, including the pre-processing transformations
# MAGIC  - generate train/test fit statistics
# MAGIC  - log model and metrics to mlflow  
# MAGIC  
# MAGIC  Although we log metrics explicitely, autologging of metrics is an option. See the [documentation here.](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging)

# COMMAND ----------

categorical_features = [col for col in features_df.columns if 'categorical' in col]
numerical_features =   [col for col in features_df.columns if 'numeric' in col]

x_train, x_test, y_train, y_test = train_test_split(features_df[categorical_features + numerical_features], 
                                                    features_df['label'], 
                                                    test_size=0.33, 
                                                    random_state=123)

preprocess_pipeline = create_preprocessing_transform(categorical_features, numerical_features)

model = XGBClassifier()

model_pipeline = Pipeline([("preprocess", preprocess_pipeline), ("classifier", model)])

# Start Mlflow logging
with mlflow.start_run() as run:
  
  run_id = run.info.run_id
  
  model_pipeline.fit(x_train, y_train)

  x_train_transformed = model_pipeline.predict(x_train)
  x_test_transformed = model_pipeline.predict(x_test) 
  
  precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, x_train_transformed, average='weighted')
  precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, x_test_transformed, average='weighted')
  
  # Capture  and log model metrics
  digits = 3
  metrics = OrderedDict()
  metrics["train_precision"]= round(precision_train, digits)
  metrics["train_recall"] =   round(recall_train, digits)
  metrics["train_f1"] =       round(f1_train, digits)
  metrics["test_precision"] = round(precision_test, digits)
  metrics["test_recall"] =    round(recall_test, digits)
  metrics["test_f1"] =        round(f1_test, digits)
  
  mlflow.log_metrics(metrics)
  
  # Create model input and output schemas
  input_schema = Schema([
    ColSpec("string", "categorical_feature_01"),
    ColSpec("double", "numeric_feature_01"),
    ColSpec("double", "numeric_feature_02"),
    ColSpec("double", "numeric_feature_03"),
    ColSpec("double", "numeric_feature_04"),
    ColSpec("double", "numeric_feature_05"),
    ColSpec("double", "numeric_feature_06"),
    ColSpec("double", "numeric_feature_07"),
    ColSpec("double", "numeric_feature_08"),
    ColSpec("double", "numeric_feature_09"),
    ColSpec("double", "numeric_feature_10"),
    ColSpec("double", "numeric_feature_11"),
    ColSpec("double", "numeric_feature_12"),
    ColSpec("double", "numeric_feature_13"),
    ColSpec("double", "numeric_feature_14"),
    ColSpec("double", "numeric_feature_15"),
    ColSpec("double", "numeric_feature_16"),
    ColSpec("double", "numeric_feature_17"),
    ColSpec("double", "numeric_feature_18"),
    ColSpec("double", "numeric_feature_19"),
    ColSpec("double", "numeric_feature_20")
  ])

  output_schema = Schema([ColSpec("integer")])

  signature = ModelSignature(inputs=input_schema, outputs=output_schema)
  
  # Log fitted model
  mlflow.sklearn.log_model(model_pipeline, signature=signature, artifact_path='model')

# COMMAND ----------

# MAGIC %md #### Create an entry in the Model Registry

# COMMAND ----------

client = MlflowClient()

model_registry_name = 'mlc_course_test'
try:
  client.get_registered_model(model_registry_name)
  print(" Registered model already exists")
except:
  client.create_registered_model(model_registry_name)
  
model_info = client.get_run(run_id).to_dictionary()
artifact_uri = model_info['info']['artifact_uri']

# Register the model
registered_model = client.create_model_version(
                     name = model_registry_name,
                     source = artifact_uri + "/model",
                     run_id = run_id
                    )

# COMMAND ----------

# MAGIC %md #### Promote the model to the production stage

# COMMAND ----------

promote_to_prod = client.transition_model_version_stage(name=model_registry_name,
                                                        version = int(registered_model.version),
                                                        stage="Production",
                                                        archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md #### Load the Production model from the registry to inference

# COMMAND ----------

def get_run_id(model_name, stage='Production'):
  """Get production model id from Model Registry"""
  
  prod_run = [run for run in client.search_model_versions(f"name='{model_name}'") 
                  if run.current_stage == stage][0]
  
  return prod_run.run_id


# Replace the first parameter with your model's name
run_id = get_run_id('mlc_course_test', stage='Production')

loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/model')

columns = categorical_features + numerical_features

predictions = loaded_model.predict(features_df[columns])

predictions

# COMMAND ----------

# MAGIC %md #### Bonus: Custom MLflow models  
# MAGIC Custom model can contain any abritrary processing logic. See the [model documentation](https://www.mlflow.org/docs/latest/models.html) and specifically the [custom models section.](https://www.mlflow.org/docs/latest/models.html#model-customization)

# COMMAND ----------

class CustomPythonModel(mlflow.pyfunc.PythonModel):
  """This custom model multiplies an input value by
  a specified multiply_by value"""
  
  def __init__(self, multiply_by):
    super().__init__()
    self.multiply_by = multiply_by
    
  def predict(self, context, model_input):
    prediction = model_input * self.multiply_by
    return prediction

# COMMAND ----------

with mlflow.start_run() as run:
  
  run_id = run.info.run_id
  print(f"run_id: {run_id}")
  
  my_custom_model = CustomPythonModel(2)
  mlflow.pyfunc.log_model(python_model=my_custom_model, artifact_path=f'model')

# COMMAND ----------

logged_model = f'runs:/{run_id}/model'
loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)

features_df['multiply_by_two'] = features_df['numeric_feature_01'].apply(lambda x: loaded_model.predict(x))

features_df[['numeric_feature_01', 'multiply_by_two']].head()
