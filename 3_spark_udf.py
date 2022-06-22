# Databricks notebook source
# MAGIC %md ## UDFs: Distributing Python functions
# MAGIC  - [A great writeup](https://medium.com/@debusinha2009/all-you-need-to-know-about-writing-custom-udf-using-python-in-apache-spark-3-0-3412a008d086) from an a Specialist Solutions Architect at Databricks
# MAGIC  - [Databricks blogs](https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html)
# MAGIC  - [Databricks docs](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html)

# COMMAND ----------

# MAGIC %pip install python-Levenshtein fuzzywuzzy

# COMMAND ----------

# MAGIC %md #### Applying a Python library to a Spark DataFrame in a distributed fashion  
# MAGIC A string similiarity example

# COMMAND ----------

from collections import OrderedDict
from typing import List, Callable
import datetime

from pyspark.sql.types import IntegerType, StructType, StringType, FloatType
import pyspark.sql.functions as func
from pyspark.sql.functions import col, pandas_udf

from fuzzywuzzy import fuzz

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.datasets import make_classification

from xgboost import XGBClassifier

# COMMAND ----------

data = [("fuzzy wuzzy was a bear", "wuzzy fuzzy was a big bear"),
        ("this is a test",         "this is a test!"),
        ("New York Giants",         "San Franciso Giants")]

columns = ["sequence_1", "sequence_2"]
df = spark.createDataFrame(data=data, schema=columns)

display(df)

# COMMAND ----------

def get_string_similiarity_func(a: pd.Series, b: pd.Series) -> pd.Series:
  """
  A Python function that accepts Pandas Series and returns a Pandas Series
  """
  
  joined = pd.concat([a, b], axis=1)
  joined['similiarity'] = joined.apply(lambda x: fuzz.token_set_ratio(x[a.name], x[b.name]), axis=1)

  return joined['similiarity']
    
# Register the function as a PandasUDF, specify the Spark data type equivalent of the
# Python function's return type
get_string_similiarity = pandas_udf(get_string_similiarity_func, returnType=IntegerType())


# Apply the PandasUDF to a Spark DataFrame
display(df.withColumn('similiarity', get_string_similiarity(col('sequence_1'), col('sequence_2'))))

# COMMAND ----------

# MAGIC %md #### PandasUDF's can scale out sophisticated operations
# MAGIC Example: training machine learning models on different groups of data in parallel using a groupBy().applyInPandas() UDF

# COMMAND ----------

# MAGIC %md  
# MAGIC 1. Create example training data that contains features for different groups of data   
# MAGIC 2. Training a separate, Python-based machine learning model on each group's data in parallel using a PandasUDF

# COMMAND ----------

groups = [[f'group_{str(n+1).zfill(2)}'] for n in range(5)]

schema = StructType()
schema.add('group_name', StringType())

df = spark.createDataFrame(groups, schema=schema)
display(df)

# COMMAND ----------

def create_group_data(group_data: pd.DataFrame) -> pd.DataFrame:

  group_name = group_data["group_name"].loc[0]
  
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

  X.ravel()[np.random.choice(X.size, missing_observations, replace=False)] = np.nan

  # Create Pandas DataFrame of numeric features
  numeric_feature_names = [f'numeric_feature_{str(n+1).zfill(2)}' for n in range(n_features)]
  df = pd.DataFrame(X, columns=numeric_feature_names)
  df['label'] = y
  df['group_name'] = group_name

  # Add a categorical column
  categories = ['A', 'B', 'C']
  cat_probs_1 = [0.7, 0.2, 0.1]
  cat_probs_0 = [0.3, 0.5, 0.2]

  df.loc[df['label'] == 1, 'categorical_feature_01'] = np.random.choice(categories, df.loc[df['label'] == 1].shape[0], p=cat_probs_1)
  df.loc[df['label'] == 0, 'categorical_feature_01'] = np.random.choice(categories, df.loc[df['label'] == 0].shape[0], p=cat_probs_0)

  col_ordering = ['group_name', 'label', 'categorical_feature_01'] + numeric_feature_names

  return df[col_ordering]

# COMMAND ----------

schema = StructType()
schema.add('group_name', StringType())
schema.add('label', IntegerType())
schema.add('categorical_feature_01', StringType())

for column_name in [f'numeric_feature_{str(n+1).zfill(2)}' for n in range(20)]:
  schema.add(column_name, FloatType())
  
features = (df.groupby('group_name').applyInPandas(create_group_data, schema=schema)
              .withColumn('id', func.monotonically_increasing_id()))

display(features)

# COMMAND ----------

# MAGIC %md Create pre-processing pipeline

# COMMAND ----------

def create_preprocessing_transform(categorical_features: List[str], numerical_features: List[str]) -> ColumnTransformer:
  
  categorical_pipe = Pipeline(
        [
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
            ("quantiles", numerical_pipe_quantile, numerical_features)
        ],
        remainder='drop'
    )

  return preprocessing

# COMMAND ----------

# MAGIC %md Configure the PandasUDF

# COMMAND ----------

def configure_model_udf(label_col: str, grouping_col:str, pipeline:ColumnTransformer, test_size:float=0.33, 
                        xgb_early_stopping_rounds:int=25, eval_metric:str="auc", random_state:int=123) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
  def train_model_udf(group_training_data: pd.DataFrame) -> pd.DataFrame:
    
    # Measure the training time of each model
    start = datetime.datetime.now()
    
    # Capture the name of the group to be modeled
    group_name = group_training_data[grouping_col].loc[0]
    
    x_train, x_test, y_train, y_test = train_test_split(group_training_data, 
                                                        group_training_data[label_col], 
                                                        test_size=test_size, 
                                                        random_state=random_state)
     
    # We must pass the testing dataset to the model to leverage early stopping,
    # and the training dataset must be transformed.
    pipeline.fit(x_train)
    x_test_transformed = pipeline.transform(x_test)
    
    # Create a scikit-learning pipeline that transforms the features and applies the
    # model
    model = XGBClassifier(n_estimators=1000)
    
    model_pipeline = Pipeline([('feature_preprocessing', pipeline),
                               ('model', model)])
    
    # Fit the model with early stopping
    # Note: Early stopping returns the model from the last iteration (not the best one). If there’s more 
    # than one item in eval_set, the last entry will be used for early stopping.
    model_pipeline.fit(x_train, y_train.values.ravel(),
                       model__eval_set = [(x_test_transformed, y_test.values.ravel())],
                       model__eval_metric=eval_metric,
                       model__early_stopping_rounds=xgb_early_stopping_rounds,
                       model__verbose=True)
    
    # Capture statistics on the best model run
    best_score = model_pipeline.named_steps['model'].best_score
    best_iteration = model_pipeline.named_steps['model'].best_iteration
    
    # Predict using only the boosters leading up to and including the best boosting 
    # round. This accounts for the fact that the model retained by xgboost is the last
    # model fit before early stopping rounds were triggered
    train_pred = model_pipeline.predict(x_train, iteration_range = (0, best_iteration + 1))
    test_pred = model_pipeline.predict(x_test, iteration_range = (0, best_iteration + 1))
    
    precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, train_pred, average='weighted')
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')
    
    end = datetime.datetime.now()
    elapsed = end-start
    seconds = round(elapsed.total_seconds(), 1)
    
    # Capture data about our the model
    digits = 3
    metrics = OrderedDict()
    metrics["train_precision"]= round(precision_train, digits)
    metrics["train_recall"] =   round(recall_train, digits)
    metrics["train_f1"] =       round(f1_train, digits)
    metrics["test_precision"] = round(precision_test, digits)
    metrics["test_recall"] =    round(recall_test, digits)
    metrics["test_f1"] =        round(f1_test, digits)
    metrics["test_auc"] =       round(best_score, digits)
    metrics["best_iteration"] = round(best_iteration, digits)
    
    other_meta = OrderedDict()
    other_meta['group'] =           group_name
    other_meta['start_time'] =      start.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    other_meta['end_time'] =        end.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    other_meta['elapsed_seconds'] = seconds
    
    other_meta.update(metrics)
    
    return pd.DataFrame(other_meta, index=[0])

  return train_model_udf

# COMMAND ----------

# MAGIC %md Apply the UDF

# COMMAND ----------

# Specify Spark DataFrame Schema
spark_types =  [('group',             StringType()),
                ('start_time',       StringType()),
                ('end_time',         StringType()),
                ('elapsed_seconds',  FloatType()),
                ('train_precision',  FloatType()),
                ('train_recall',     FloatType()),
                ('train_f1',         FloatType()),
                ('test_precision',   FloatType()),
                ('test_recall',      FloatType()),
                ('test_f1',          FloatType()),
                ('test_auc',         FloatType()),
                ('best_iteration',   IntegerType())]

spark_schema = StructType()
for col_name, spark_type in spark_types:
  spark_schema.add(col_name, spark_type)
  
  
categorical_features = [col for col in features.columns if 'categorical' in col]
numerical_features =   [col for col in features.columns if 'numeric' in col]
label_col =            ['label']
grouping_col =         'group_name'

# Create a pre-processing pipeline instance
pipeline = create_preprocessing_transform(categorical_features, numerical_features)

# Configure the PandasUDF
train_model_udf = configure_model_udf(label_col, 
                                      grouping_col, 
                                      pipeline)

best_model_stats = features.groupBy('group_name').applyInPandas(train_model_udf, schema=spark_schema)

display(best_model_stats)
