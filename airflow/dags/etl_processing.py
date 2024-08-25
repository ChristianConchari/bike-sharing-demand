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
    requirements=["ucimlrepo==0.0.7", "awswrangler==3.9.1"],
    system_site_packages=True
    )
    def get_data() -> None:
        """
        Load the original dataset from UCI ML Repository
        """
        import awswrangler as wr
        import pandas as pd
        from ucimlrepo import fetch_ucirepo
        from airflow.models import Variable
        
        # Fetch the dataset from UCI ML Repository
        bike_sharing_demand = fetch_ucirepo(id=275)
        df_features = bike_sharing_demand.data.features
        df_targets = bike_sharing_demand.data.targets

        # Join the features and target DataFrames
        df = pd.concat([df_features, df_targets], axis=1)
        
        # Save the original dataset to S3
        data_path = 's3://data/raw/bike_sharing_demand.csv'
        wr.s3.to_csv(df, data_path, index=False)
        
    @task.virtualenv(
    task_id='feature_engineering',
    requirements=["awswrangler==3.9.1"],
    system_site_packages=True
    )
    def feature_engineering() -> None:
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

        def one_hot_encode(df: pd.DataFrame, one_hot_cols: list) -> pd.DataFrame:
            """
            Applies One-Hot Encoding to specified columns of a DataFrame.
            
            Parameters:
            - df (pd.DataFrame): The input DataFrame.
            - one_hot_cols (list): List of column names to apply One-Hot Encoding to.
            
            Returns:
            - pd.DataFrame: The DataFrame with One-Hot Encoded columns.
            """
            # Apply One-Hot Encoding
            df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)
            return df

        def cyclic_encode(df: pd.DataFrame, columns: list, max_value: int) -> pd.DataFrame:
            """
            Applies cyclic encoding to specified columns of a DataFrame.
            
            Parameters:
            - df (pd.DataFrame): The input DataFrame.
            - columns (list): List of column names to apply Cyclic Encoding to.
            - max_value (int): The maximum value the cyclic variable can take (e.g., 23 for hours).
            
            Returns:
            - pd.DataFrame: The DataFrame with Cyclic Encoded columns added.
            """
            for col_name in columns:
                # Compute the sin and cos components
                df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_value)
                df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_value)
                # Drop the original column
                df = df.drop(col_name, axis=1)
                
            return df
        
        # Set paths for the original and processed data
        data_original_path = 's3://data/raw/bike_sharing_demand.csv'
        data_processed_path = 's3://data/processed/bike_sharing_demand.csv'
        
        # Read the original dataset from S3
        df = wr.s3.read_csv(data_original_path)
        
        # Drop duplicates and null values
        df = df.drop_duplicates()
        df = df.dropna()
        
        # Normalize count column by taking the log
        df['log_cnt'] = np.log(df['cnt'])
        
        # Calculate the correlation matrix
        corr = df.corr()
        
        # Find pairs of columns with high correlation
        high_corr_pairs = [
            (col1, col2) for col1 in corr.columns for col2 in corr.columns 
            if col1 != col2 and abs(corr.loc[col1, col2]) > 0.85 
        ]
        
        # Mantain only selected features
        selected_columns = Variable.get("features")
        df = df[selected_columns + ['log_cnt']]

        # Define lists for tracking encoded columns
        one_hot_encoded_columns = ['weathersit', 'season', 'weekday']
        cyclic_encoded_columns = ['hr']
        
        df_encoded = df.copy()

        # Encode using one-hot encoding
        df_encoded = one_hot_encode(df_encoded, one_hot_encoded_columns)
        # Encode using cyclic encoding
        df_encoded = cyclic_encode(df_encoded, cyclic_encoded_columns)
        
        # Save the processed dataset to S3
        wr.s3.to_csv(df_encoded, data_processed_path, index=False)

        # Save information about the dataset
        client = boto3.client('s3')
        data_dict = {}
        
        try:
            client.head_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
            result = client.get_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                raise e

        # Get the target column and drop it from the dataset
        target_col = Variable.get("target")
        dataset_log = df.drop(columns=target_col)
        dataset_encoded_log = df_encoded.drop(columns=target_col)
        
        # Save information about the dataset
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_encoded'] = dataset_encoded_log.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = one_hot_encoded_columns + cyclic_encoded_columns
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_encoded_log.dtypes.to_dict().items()}

        category_encoded_dict = {}
        
        # Save the categories for each categorical column
        for category in one_hot_encoded_columns + cyclic_encoded_columns:
            category_encoded_dict[category] = np.sort(dataset_log[category].unique()).tolist()

        data_dict['categories_values_per_categorical'] = category_dummies_dict

        # Track the date and time the data was processed
        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')
        data_string = json.dumps(data_dict, indent=2)

        # Save the data dictionary to S3
        client.put_object(
            Bucket='data',
            Key='data_info/bike_sharing_demand_data_info.json',
            Body=data_string
        )

        # Log the data dictionary to MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Bike Sharing Demand")

        # Start a new MLflow run
        mlflow.start_run(
            run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "Bike Sharing"},
            log_system_metrics=True
        )
        
        # Log the complete data to MLflow
        mlflow_dataset = mlflow.data.from_pandas(
                            df,
                            source="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
                            targets=target_col,
                            name="bike_sharing_complete"
                        )

        # Log the encoded data to MLflow
        mlflow_dataset_encoded = mlflow.data.from_pandas(
                                    df_encoded,
                                    source="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
                                    targets=target_col,
                                    name="bike_sharing_processed_encoded"
                                )
        
        # Log the data to MLflow
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_encoded, context="Dataset")

        # End the MLflow run
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
        data_processed_path = 's3://data/processed/bike_sharing_demand.csv'
        
        # Read the processed dataset from S3
        data = wr.s3.read_csv(data_processed_path)
        
        # Get the target column
        target_col = Variable.get("target")
        
        # Define features and target
        X = data.drop(columns=target_col, axis=1)
        y = data[target_col]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Variable.get("test_size_bike"), random_state=42)
        
        # Save the training and testing datasets to S3
        X_train_path = 's3://data/train/bike_sharing_demand_X_train.csv'
        X_test_path = 's3://data/test/bike_sharing_demand_X_test.csv'
        y_train_path = 's3://data/train/bike_sharing_demand_y_train.csv'
        y_test_path = 's3://data/test/bike_sharing_demand_y_test.csv'
        
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
        
        # Read the training and testing datasets from S3
        X_train_path = 's3://data/train/bike_sharing_demand_X_train.csv'
        X_test_path = 's3://data/test/bike_sharing_demand_X_test.csv'
        
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
        X_train_scaled_path = 's3://data/train/bike_sharing_demand_X_train_scaled.csv'
        X_test_scaled_path = 's3://data/test/bike_sharing_demand_X_test_scaled.csv'
        
        wr.s3.to_csv(X_train_scaled, X_train_scaled_path, index=False)
        wr.s3.to_csv(X_test_scaled, X_test_scaled_path, index=False)
        
        # Save information about the scaler
        client = boto3.client('s3')
        
        try:
            client.head_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
            result = client.get_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
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
            Bucket='data',
            Key='data_info/bike_sharing_demand_data_info.json',
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
        