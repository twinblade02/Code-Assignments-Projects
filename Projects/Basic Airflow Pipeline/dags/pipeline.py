from email.policy import default
import os
from airflow import DAG
from datetime import datetime, timedelta
import csv
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.apache.hive.operators.hive import HiveOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

default_args = {
    "owner": "airflow",
    "email_on_failure": False,
    "email_on_retry": False,
    "email": "###",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

def walk_directory(filepath):
    files = []
    #names = []
    for pth, direc, fle in os.walk(filepath):
        for f in fle:
            if f.endswith(".csv"):
                files.append(os.path.join(pth,f))
                #names.append(f)
    return files


def uploads3(filename: str, key: str, bucket_name: str) -> None:
    hook = S3Hook('s3_conn')
    for f in walk_directory(filename):
        hook.load_file(filename=f, key=f, bucket_name=bucket_name)
        print(f"File {f} upload complete")


with DAG('pipeline', start_date=datetime(2022,3, 28), schedule_interval='@daily', default_args=default_args, catchup=False) as dag:

    is_csv_available = FileSensor(
        task_id='is_csv_available',
        fs_conn_id = 'path',
        filepath="owid-covid-data.csv",
        poke_interval=5,
        timeout=20
    )
    
    push_to_hive = BashOperator(
        task_id = "push_to_hive",
        bash_command = """
            hdfs dfs -mkdir -p /covidData && \
            hdfs dfs -put -f $AIRFLOW_HOME/dags/files/owid-covid-data.csv /covidData
        """
    )

    create_hive_table = HiveOperator(
        task_id = 'create_hive_table',
        hive_cli_conn_id = 'hive_conn',
        hql = """
            CREATE EXTERNAL TABLE IF NOT EXISTS cov_data(
                iso_code STRING,
                continent STRING,
                location STRING,
                `date` STRING,
                total_cases BIGINT,
                new_cases BIGINT,
                new_cases_smoothed FLOAT,
                total_deaths BIGINT,
                new_deaths BIGINT,
                new_deaths_smoothed INT,
                total_cases_per_million BIGINT,
                new_cases_per_million FLOAT,
                new_cases_smoothed_per_million FLOAT,
                total_deaths_per_million FLOAT,
                new_deaths_per_million FLOAT,
                new_deaths_smoothed_per_million FLOAT,
                reproduction_rate FLOAT,
                icu_patients INT,
                icu_patients_per_million FLOAT,
                hosp_patients BIGINT,
                hosp_patients_per_million FLOAT,
                weekly_icu_admissions INT,
                weekly_icu_admissions_per_million FLOAT,
                weekly_hosp_admissions BIGINT,
                weekly_hosp_admissions_per_million FLOAT,
                total_tests BIGINT,
                new_tests BIGINT,
                total_tests_per_thousand FLOAT,
                new_tests_per_thousand FLOAT,
                new_tests_smoothed BIGINT,
                new_tests_smoothed_per_thousand FLOAT,
                positive_rate FLOAT,
                tests_per_case BIGINT,
                tests_units STRING,
                total_vaccinations BIGINT,
                people_vaccinated BIGINT,
                people_fully_vaccinated BIGINT,
                total_boosters BIGINT,
                new_vaccinations BIGINT,
                new_vaccinations_smoothed BIGINT,
                total_vaccinations_per_hundred INT,
                people_vaccinated_per_hundred INT,
                people_fully_vaccinated_per_hundred INT,
                total_boosters_per_hundred INT,
                new_vaccinations_smoothed_per_million INT,
                new_people_vaccinated_smoothed BIGINT,
                new_people_vaccinated_smoothed_per_hundred FLOAT,
                stringency_index INT,
                population BIGINT,
                population_density BIGINT,
                median_age FLOAT,
                aged_65_older FLOAT,
                aged_70_older FLOAT,
                gdp_per_capita BIGINT,
                extreme_poverty INT,
                cardiovasc_death_rate FLOAT,
                diabetes_prevalence FLOAT,
                female_smokers FLOAT,
                male_smokers FLOAT,
                handwashing_facilities INT,
                hospital_beds_per_thousand INT,
                life_expectancy FLOAT,
                human_development_index FLOAT,
                excess_mortality_cumulative_absolute BIGINT,
                excess_mortality_cumulative FLOAT,
                excess_mortality INT,
                excess_mortality_cumulative_per_million BIGINT 
                )
            COMMENT 'Main Table'
            ROW FORMAT DELIMITED
            FIELDS TERMINATED BY ','
            TBLPROPERTIES ("skip.header.line.count"="1");
        """
    )

    populate_hive_table = HiveOperator(
        task_id = 'populate_hive_table',
        hive_cli_conn_id = 'hive_conn',
        hql = """
            LOAD DATA INPATH '/covidData/owid-covid-data.csv' INTO TABLE cov_data
        """
    )

    processing = SparkSubmitOperator(
        task_id = 'processing',
        application = "/opt/airflow/dags/scripts/spark_processing.py",
        conn_id = 'spark_conn',
        verbose = True
    )

    copy_to_files = BashOperator(
        task_id = 'copy_to_files',
        bash_command = """
        hdfs dfs -get -f /IndiaCOVID /opt/airflow/dags/files
        """
    )

    task_uploads3 = PythonOperator(
        task_id = 'task_uploads3',
        python_callable = uploads3,
        op_kwargs = {
            'filename': '/opt/airflow/dags/files/IndiaCOVID',
            'key': 'f',
            'bucket_name': 'c19-backups-airflow'
        }
    )

    is_csv_available >> push_to_hive >> create_hive_table >> populate_hive_table >> processing >> copy_to_files >> task_uploads3