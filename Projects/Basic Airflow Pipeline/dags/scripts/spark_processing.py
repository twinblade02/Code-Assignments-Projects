from os.path import abspath
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession

warehouse_location = abspath('spark-warehouse')
hdfs_dir = 'hdfs://namenode:9000/IndiaCOVID'
new_filename = 'ICOVID.csv'

# spark session
spark = SparkSession.builder.appName('Processing').config('spark.sql.warehouse.dir', warehouse_location) \
    .enableHiveSupport().getOrCreate()

# get table
run_hql_Ind = "SELECT * FROM default.cov_data WHERE location = 'India'"
table_Ind = spark.sql(run_hql_Ind)

# save this partitioned table to HDFS namenode and local filesystem
table_Ind.write.option('header','true').csv('hdfs://namenode:9000/IndiaCOVID')
