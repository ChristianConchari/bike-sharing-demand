"""
This DAG extracts data from a source CSV file, transforms the data, and loads it into a destination S3 bucket
as two separate CSV files, one for training and one for testing. The split between training and testing data is 70/30.
"""
from datetime import timedelta
from airflow.decorators import dag, task

MARKDOWN_TEXT = """
# ETL Pipeline

This DAG extracts data from a source CSV file, transforms the data, and loads it into a destination S3 bucket
as two separate CSV files, one for training and one for testing. The split between training and testing data is 70/30.
"""

default_args = {
    'owner': 'Christian Conchari',
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=15)
}

@dag(
    dag_id='process_etl_bike_sharing_data',
    description='ETL process for bike sharing demand data, splitting data into training and testing datasets',
    doc_md=MARKDOWN_TEXT,
    tags=['etl', 'bike_sharing_demand'],
    default_args=default_args,
    catchup=False
)
def etl_processing():
    """
    ETL process for bike sharing demand data, splitting data into training and testing datasets
    """
    @task.virtualenv(
    task_id='get_original_data',
    requirements=["awswrangler==3.9.1"],
    system_site_packages=True
    )
    def get_data():
        """
        Load the original dataset from a CSV file and save it to an S3 bucket
        """
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        
        # Obtain the original dataset
        df = pd.read_csv('/opt/airflow/data/train.csv')
        
        # Save the original dataset to S3
        data_path = 's3://mlflow/data/raw/bike_sharing_raw.csv'
        wr.s3.to_csv(df, data_path, index=False)
        
    @task.virtualenv(
    task_id='feature_engineering',
    requirements=["awswrangler==3.9.1"],
    system_site_packages=True
    )
    def feature_engineering():
        """
        Perform feature engineering on the dataset
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow
        import os
        import sys

        import awswrangler as wr
        import pandas as pd
        import numpy as np

        from airflow.models import Variable
        from sklearn.preprocessing import LabelEncoder

        def label_encode(df, columns):
            """
            Applies Label Encoding to specified columns of a DataFrame.
            
            Parameters:
            df (pd.DataFrame): The input DataFrame.
            columns (list): List of column names to apply Label Encoding to.
            
            Returns:
            pd.DataFrame: The DataFrame with the Label Encoded columns.
            """
            le = LabelEncoder()
            
            for col_label in columns:
                # Apply Label Encoding
                df[f'{col_label}_label'] = le.fit_transform(df[col_label])
            
            # Drop the original columns
            df = df.drop(columns, axis=1)
            
            return df

        def one_hot_encode(df, one_hot_cols):
            """
            Applies One-Hot Encoding to specified columns of a DataFrame.
            
            Parameters:
            df (pd.DataFrame): The input DataFrame.
            one_hot_cols (list): List of column names to apply One-Hot Encoding to.
            
            Returns:
            pd.DataFrame: The DataFrame with One-Hot Encoded columns.
            """
            # Apply One-Hot Encoding
            df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)
            return df

        def cyclic_encode(df, columns, max_value=23):
            """
            Applies cyclic encoding to specified columns of a DataFrame.
            
            Parameters:
            df (pd.DataFrame): The input DataFrame.
            columns (list): List of column names to apply Cyclic Encoding to.
            max_value (int): The maximum value the cyclic variable can take (e.g., 23 for hours).
            
            Returns:
            pd.DataFrame: The DataFrame with Cyclic Encoded columns added.
            """
            for col_name in columns:
                # Compute the sin and cos components
                df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_value)
                df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_value)
                # Drop the original column
                df = df.drop(col_name, axis=1)
                
            return df
        
        # Set paths for the original and processed data
        data_original_path = 's3://mlflow/data/raw/bike_sharing_raw.csv'
        data_processed_path = 's3://mlflow/data/processed/bike_sharing_processed.csv'
        
        # Read the original dataset from S3
        df = wr.s3.read_csv(data_original_path)
        
        # Drop duplicates and null values
        df = df.drop_duplicates()
        df = df.dropna()
        
        # Ensure the datetime column is in the correct format
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract year, month, weekday, weekend, and hour
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['hour'] = df['datetime'].dt.hour
        
        # Drop the datetime column
        df = df.drop(columns=['datetime'], axis=1)
        
        # Normalize count column by taking the log
        df['log_count'] = np.log(df['count'] + 1)
        df['log_casual'] = np.log(df['casual'] + 1)
        df['log_registered'] = np.log(df['registered'] + 1)
        
        # Drop the original count, casual, and registered columns
        df = df.drop(['count', 'casual', 'registered'], axis=1)
        
        # Calculate the correlation matrix
        corr = df.corr()
        
        # Find pairs of columns with high correlation
        high_corr_pairs = [
            (col1, col2) for col1 in corr.columns for col2 in corr.columns 
            if col1 != col2 and abs(corr.loc[col1, col2]) > 0.85 
        ]
        
        # Identify columns to drop
        to_drop = set()

        for col1, col2 in high_corr_pairs:
            if abs(corr.loc[col1, 'log_count']) > abs(corr.loc[col2, 'log_count']):
                to_drop.add(col2)
            else:
                to_drop.add(col1)

        # Drop the identified columns
        df = df.drop(columns=to_drop)
        
        # Tracking categorical columns before encoding
        original_categorical_columns = ['holiday', 'workingday', 'year', 'weather', 'month', 'weekday', 'hour']

        # Define lists for tracking encoded columns
        label_encoded_columns = ['holiday', 'workingday', 'year']
        one_hot_encoded_columns = ['weather', 'month', 'weekday']
        cyclic_encoded_columns = ['hour']

        # Encode using label encoding
        df = label_encode(df, label_encoded_columns)
        # Encode using one-hot encoding
        df = one_hot_encode(df, one_hot_encoded_columns)
        # Encode using cyclic encoding
        df = cyclic_encode(df, cyclic_encoded_columns)
        
        # Save the processed dataset to S3
        wr.s3.to_csv(df, data_processed_path, index=False)

        client = boto3.client('s3')
        data_dict = {}
        
        try:
            client.head_object(Bucket='mlflow', Key='data_info/bike_sharing_data_info.json')
            result = client.get_object(Bucket='mlflow', Key='data_info/bike_sharing_data_info.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                raise e

        target_col = 'log_count'
        dataset_log = df.drop(columns=target_col)

        # Save information about the dataset
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['target_col'] = target_col
        
        # Track original categorical columns
        data_dict['original_categorical_columns'] = original_categorical_columns

        # Tracking details of label encoding, one-hot encoding, and cyclic encoding
        label_encoded_dict = {}
        one_hot_encoded_dict = {}
        cyclic_encoded_dict = {}

        # Tracking unique values for label encoded columns
        for col in label_encoded_columns:
            label_encoded_dict[col] = df[f'{col}_label'].unique().tolist()

        # Tracking one-hot encoded columns and their resulting dummy variables
        for col in one_hot_encoded_columns:
            one_hot_encoded_dict[col] = df.filter(like=f'{col}_').columns.to_list()

        # Tracking original values and transformed values for cyclically encoded columns
        for col in cyclic_encoded_columns:
            cyclic_encoded_dict[col] = {
                'original_values': df[[f'{col}_sin', f'{col}_cos']].drop_duplicates().index.tolist(),
                'transformed_columns': [f'{col}_sin', f'{col}_cos']
            }

        # Adding these details to the data dictionary
        data_dict['label_encoded_columns'] = label_encoded_dict
        data_dict['one_hot_encoded_columns'] = one_hot_encoded_dict
        data_dict['cyclic_encoded_columns'] = cyclic_encoded_dict

        # Track the date and time the data was processed
        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')
        data_string = json.dumps(data_dict, indent=2)

        # Save the data dictionary to S3
        client.put_object(
            Bucket='mlflow',
            Key='data_info/bike_sharing_data_info.json',
            Body=data_string
        )

        # Log the data dictionary to MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Bike Sharing Demand")

        # Start a new MLflow run
        mlflow.start_run(run_name='Feature_Engineering_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
                        experiment_id=experiment.experiment_id,
                        tags={"experiment": "feature_engineering", "dataset": "Bike Sharing"},
                        log_system_metrics=True)
        
        # Log the processed dataset to MLflow
        mlflow_dataset = mlflow.data.from_pandas(df,
                                                source="s3://mlflow/data/raw/bike_sharing_raw.csv",
                                                targets=target_col,
                                                name="bike_sharing_processed")
        
        # Log the data to MLflow
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_artifact(data_processed_path, artifact_path="processed_data")
        mlflow.log_dict(data_dict, "bike_sharing_data_info.json")

        mlflow.end_run()

    
    @task.virtualenv(
        task_id='split_dataset',
        requirements=[ "awswrangler==3.9.1" ],
        system_site_packages=True
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable
        
        # Processed dataset path
        data_processed_path = 's3://mlflow/data/processed/bike_sharing_processed.csv'
        
        # Read the processed dataset from S3
        data = wr.s3.read_csv(data_processed_path)
        
        # Define X variables and target variable
        X = data.drop(columns=['log_count'], axis=1)
        y = data['log_count']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Save the training and testing datasets to S3
        X_train_path = 's3://mlflow/data/train/bike_sharing_X_train.csv'
        X_test_path = 's3://mlflow/data/test/bike_sharing_X_test.csv'
        y_train_path = 's3://mlflow/data/train/bike_sharing_y_train.csv'
        y_test_path = 's3://mlflow/data/test/bike_sharing_y_test.csv'
        
        wr.s3.to_csv(X_train, X_train_path, index=False)
        wr.s3.to_csv(X_test, X_test_path, index=False)
        wr.s3.to_csv(y_train, y_train_path, index=False)
        wr.s3.to_csv(y_test, y_test_path, index=False)

    @task.virtualenv(
        task_id='normalize_data',
        requirements=[ "awswrangler==3.9.1" ],
        system_site_packages=True
    )
    def normalize_data():
        import json
        import mlflow
        import boto3
        import botocore.exceptions
        
        import pandas as pd
        import awswrangler as wr
        
        from sklearn.preprocessing import StandardScaler
        
        # Save the training and testing datasets to S3
        X_train_path = 's3://mlflow/data/train/bike_sharing_X_train.csv'
        X_test_path = 's3://mlflow/data/test/bike_sharing_X_test.csv'
        
        # Read the training and testing datasets from S3
        X_train = wr.s3.read_csv(X_train_path)
        X_test = wr.s3.read_csv(X_test_path)

        # Initialize the scaler
        scaler = StandardScaler(with_mean=True, with_std=True)
        
        # Fit the scaler on the training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Conver to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Save the scaled training and testing datasets to S3
        X_train_scaled_path = 's3://mlflow/data/train/bike_sharing_X_train_scaled.csv'
        X_test_scaled_path = 's3://mlflow/data/test/bike_sharing_X_test_scaled.csv'
        
        wr.s3.to_csv(X_train_scaled, X_train_scaled_path, index=False)
        wr.s3.to_csv(X_test_scaled, X_test_scaled_path, index=False)
        
        # Save information about the scaler
        client = boto3.client('s3')
        
        try:
            client.head_object(Bucket='mlflow', Key='data_info/bike_sharing_data_info.json')
            result = client.get_object(Bucket='mlflow', Key='data_info/bike_sharing_data_info.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                raise e

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = scaler.mean_.tolist()
        data_dict['standard_scaler_std'] = scaler.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='mlflow',
            Key='data_info/bike_sharing_data_info.json',
            Body=data_string
        )

        # Log the data dictionary to MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Bike Sharing Demand")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train_scaled.shape[0])
            mlflow.log_param("Test observations", X_test_scaled.shape[0])
            mlflow.log_param("Standard Scaler feature names", scaler.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", scaler.mean_)
            mlflow.log_param("Standard Scaler scale values", scaler.scale_)
    
    get_data() >> feature_engineering() >> split_dataset() >> normalize_data()
    
dag = etl_processing()
        