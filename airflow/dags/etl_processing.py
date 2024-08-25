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
    ETL process for bike sharing demand data, extracting data from a source CSV file, 
    transforming the data, and loading it into a destination S3 bucket as two separate CSV files,
    """
    @task.virtualenv(
        task_id='get_original_data',
        requirements=["ucimlrepo==0.0.7", "awswrangler==3.9.1"],
        system_site_packages=True
    )
    def get_data() -> None:
        """
        Load the original dataset from UCI ML Repository and save it to S3.
        """
        # Import necessary libraries
        import logging
        import awswrangler as wr
        import pandas as pd
        from ucimlrepo import fetch_ucirepo

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Define the S3 path where the dataset will be saved
        data_path = 's3://data/raw/bike_sharing_demand.csv'

        logger.info("Starting to fetch the dataset from UCI ML Repository")

        try:
            # Fetch the dataset from UCI ML Repository
            bike_sharing_demand = fetch_ucirepo(id=275)
            df_features = bike_sharing_demand.data.features
            df_targets = bike_sharing_demand.data.targets
            logger.info("Dataset fetched successfully from UCI ML Repository")
        except Exception as e:
            logger.error(f"Failed to fetch dataset from UCI ML Repository: {e}")
            raise

        try:
            # Join the features and target DataFrames
            df = pd.concat([df_features, df_targets], axis=1)
            logger.info("Features and targets DataFrames concatenated successfully")
        except Exception as e:
            logger.error(f"Failed to concatenate features and targets DataFrames: {e}")
            raise

        logger.info(f"Starting to save the dataset to S3 at {data_path}")

        try:
            # Save the original dataset to S3
            wr.s3.to_csv(df, data_path, index=False)
            logger.info(f"Dataset saved successfully to {data_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset to S3 at {data_path}: {e}")
            raise
        
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
        import logging
        import boto3
        import botocore.exceptions
        import mlflow
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from airflow.models import Variable
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        def one_hot_encode(df: pd.DataFrame, one_hot_cols: list) -> pd.DataFrame:
            """
            Apply One-Hot Encoding to specified columns of a DataFrame.

            Parameters:
            - df (pd.DataFrame): The input DataFrame.
            - one_hot_cols (list): List of column names to apply One-Hot Encoding to.

            Returns:
            - pd.DataFrame: The DataFrame with One-Hot Encoded columns.
            """
            logger.info(f"Applying One-Hot Encoding to columns: {one_hot_cols}")
            # Apply One-Hot Encoding and return modified DataFrame
            df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)
            return df

        def cyclic_encode(df: pd.DataFrame, columns: list, max_value: int = 23) -> pd.DataFrame:
            """
            Apply cyclic encoding to specified columns of a DataFrame.

            Parameters:
            - df (pd.DataFrame): The input DataFrame.
            - columns (list): List of column names to apply Cyclic Encoding to.
            - max_value (int): The maximum value the cyclic variable can take (e.g., 23 for hours).

            Returns:
            - pd.DataFrame: The DataFrame with Cyclic Encoded columns added.
            """
            logger.info(f"Applying Cyclic Encoding to columns: {columns} with max value {max_value}")
            for col_name in columns:
                # Compute sin and cos components for cyclic encoding
                df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_value)
                df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_value)
                # Drop the original column after encoding
                df = df.drop(col_name, axis=1)
            return df
        
        
        # Load Dataset
        logger.info("Loading dataset from S3")
        # Define S3 paths for original and processed data
        data_original_path = 's3://data/raw/bike_sharing_demand.csv'
        data_processed_path = 's3://data/processed/bike_sharing_demand.csv'
        
        try:
            # Read the original dataset from S3
            df = wr.s3.read_csv(data_original_path)
            logger.info(f"Dataset loaded successfully from {data_original_path}")
        except Exception as e:  
            logger.error(f"Failed to load dataset from {data_original_path}: {e}")
            raise
        
        # Data Cleaning
        logger.info("Cleaning data by removing duplicates and null values")
        # Remove duplicates and null values from the dataset
        df = df.drop_duplicates()
        df = df.dropna()
        
        # Feature Engineering
        logger.info("Starting feature engineering")
        
        # Normalize the 'cnt' column by taking the log
        df['log_cnt'] = np.log(df['cnt'])
        logger.info("Normalized 'cnt' column by applying log transformation")
        
        # Select features to retain in the processed dataset
        selected_columns = ['season', 'yr', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
        df = df[selected_columns + ['log_cnt']]
        logger.info(f"Selected columns for processing: {selected_columns + ['log_cnt']}")

        # Lists of columns to be encoded
        one_hot_encoded_columns = ['weathersit', 'season', 'weekday']
        cyclic_encoded_columns = ['hr']
        
        # Copy the DataFrame to avoid modifying the original
        df_encoded = df.copy()

        # Apply One-Hot Encoding
        df_encoded = one_hot_encode(df_encoded, one_hot_encoded_columns)
        # Apply Cyclic Encoding
        df_encoded = cyclic_encode(df_encoded, cyclic_encoded_columns)
        
        logger.info("Feature engineering completed")
        
        # Save Processed Data
        logger.info(f"Saving processed dataset to {data_processed_path}")
        # Save the processed dataset to S3
        try:
            wr.s3.to_csv(df_encoded, data_processed_path, index=False)
            logger.info(f"Processed dataset saved successfully to {data_processed_path}")
        except Exception as e:
            logger.error(f"Failed to save processed dataset to {data_processed_path}: {e}")
            raise

        # Update Dataset Information
        logger.info("Updating dataset information in S3")
        # Initialize S3 client to manage dataset information
        s3_client = boto3.client('s3')
        data_dict = {}
        
        # Attempt to fetch existing dataset information
        try:
            s3_client.head_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
            result = s3_client.get_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
            logger.info("Existing dataset information loaded from S3")
        except botocore.exceptions.ClientError as e:
            # If the object is not found, continue with an empty dictionary
            if e.response['Error']['Code'] == "404":
                logger.info("No existing dataset information found, initializing new info dictionary")
            else:
                logger.error(f"Failed to fetch dataset information: {e}")
                raise

        # Fetch target column from Airflow Variable
        target_col = Variable.get("target_col")
        logger.info(f"Target column for dataset: {target_col}")
        
        # Prepare data information for logging
        dataset_log = df.drop(columns=target_col)
        dataset_encoded_log = df_encoded.drop(columns=target_col)
        
        # Update data dictionary with dataset details
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_encoded'] = dataset_encoded_log.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = one_hot_encoded_columns + cyclic_encoded_columns
        data_dict['one_hot_encoded_columns'] = one_hot_encoded_columns
        data_dict['cyclic_encoded_columns'] = cyclic_encoded_columns
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_encoded_log.dtypes.to_dict().items()}

        # Save the unique values of categorical columns for reference
        category_encoded_dict = {}
        for category in one_hot_encoded_columns + cyclic_encoded_columns:
            category_encoded_dict[category] = np.sort(dataset_log[category].unique()).tolist()
        data_dict['categories_values_per_categorical'] = category_encoded_dict

        # Add the current date and time to the data dictionary
        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')
        
        # Convert data dictionary to JSON string
        data_string = json.dumps(data_dict, indent=2)

        # Save updated data dictionary back to S3
        try:
            s3_client.put_object(
                Bucket='data',
                Key='data_info/bike_sharing_demand_data_info.json',
                Body=data_string
            )
            logger.info("Dataset information updated successfully in S3")
        except Exception as e:
            logger.error(f"Failed to update dataset information in S3: {e}")
            raise
        
        # Log Data to MLflow
        logger.info("Logging data to MLflow")

        # Configure MLflow tracking server
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Bike Sharing Demand")

        # Start a new MLflow run
        mlflow.start_run(
            run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "Bike Sharing"},
            log_system_metrics=True
        )
        
        # Log original and processed datasets to MLflow
        mlflow_dataset = mlflow.data.from_pandas(
                            df,
                            source="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
                            targets=target_col,
                            name="bike_sharing_complete"
                        )
        mlflow_dataset_encoded = mlflow.data.from_pandas(
                                    df_encoded,
                                    source="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
                                    targets=target_col,
                                    name="bike_sharing_processed_encoded"
                                )
        
        # Log datasets to MLflow
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_encoded, context="Dataset")
        logger.info("Datasets logged to MLflow successfully")

        # End the MLflow run
        mlflow.end_run()
        logger.info("MLflow run ended")

    
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
        target_col = Variable.get("target_col")
        
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
        