from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

import numpy as np
import pandas as pd


spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


def get_current_user():
  """Get the current notebook user"""
  return dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0].replace('.', '_')

#def get_data_location():
#  return 'dbfs:/Filestore/course_data/car_insurance_claim.csv'


def get_database_name(database):
  """Return a database name string based on user initials"""
  if len(database.split()) != 1:
    raise ValueError("Requires valid initials")
  
  return  f"workshop_{database}"


def list_files(table_dir):
  """View underying table files"""
  
  files = dbutils.fs.ls(table_dir)
  file_names = [(file.name, file.size) for file in files]

  for index, file_data in enumerate(file_names):
    if file_data[0].find('parquet') > -1:
      print(f"{index} - size: {file_data[1]} - {file_data[0]}")
      
      
def get_num_files(table_dir):
  """Given a director, return the number of parquet files"""
  
  files = dbutils.fs.ls(table_dir)
  parquet_files = [file for file in files if file.name.find('parquet') > -1]
  
  number_of_files = len(parquet_files)
  
  print("Number of files: {}".format(number_of_files))
  
  
def bytes_to_megabytes(bytes):
  """Return number of bytes in megabytes"""
  
  megabytes = bytes * 0.000001
  print(f"Megabytes: {round(megabytes, 2)}")
  
  
def megabytes_to_bytes(megabytes):
  """Return number of bytes in megabytes"""
  
  megabytes = megabytes / 0.000001
  print(f"Bytes: {megabytes}")