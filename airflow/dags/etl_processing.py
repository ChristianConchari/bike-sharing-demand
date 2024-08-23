"""
This DAG extracts data from a source CSV file, transforms the data, and loads it into a destination S3 bucket
as two separate CSV files, one for training and one for testing. The split between training and testing data is 70/30.
"""
from datetime import timedelta
from airflow.decorators import dag, task
import pandas as pd
import awswrangler as wr
import os
import sys

# Get the absolute path of the src directory
src_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
# Add the src directory to the Python path
sys.path.append(src_path)

# Now you can import your functions
from encoding_functions import cyclic_encode, label_encode, one_hot_encode

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
    @task()
    def get_data():
        """
        Load the original dataset from a CSV file and save it to an S3 bucket
        """
        # Obtain the original dataset
        df = pd.read_csv('data/train.csv')
        
        # Save the original dataset to S3
        data_path = 's3://mlflow/data/raw/bike_sharing_raw.csv'
        wr.s3.to_csv(df, data_path, index=False)
        
    @task.virtualenv()
    def feature_engineering():
        """
        Perform feature engineering on the dataset
        """
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
            # Pair of columns with correlation value
            (col1, col2) for col1 in corr.columns for col2 in corr.columns 
            # Ensure the pair is unique and the correlation value is greater than 0.85
            if col1 != col2 and abs(corr.loc[col1, col2]) > 0.85 
        ]
        
        # Identify columns to drop
        to_drop = set()

        for col1, col2 in high_corr_pairs:
            # Drop the column with the least correlation with the target
            if abs(corr.loc[col1, 'log_count']) > abs(corr.loc[col2, 'log_count']):
                to_drop.add(col2)
            else:
                to_drop.add(col1)

        # Drop the identified columns
        df = df.drop(columns=to_drop)
        
        # Tracking categorical columns before encoding
        original_categorical_columns = ['holiday', 'workingday', 'year', 'weather', 'month', 'weekday', 'hour']

        # Encode using label encoding
        df = label_encode(df, ['holiday', 'workingday', 'year'])
        # Encode using one-hot encoding
        df = one_hot_encode(df, ['weather', 'month', 'weekday'])
        # Encode using cyclic encoding
        df = cyclic_encode(df, ['hour'])
        
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

        # Track encoded columns
        data_dict['label_encoded_columns'] = label_encoded_columns
        data_dict['one_hot_encoded_columns'] = {col: df.filter(like=f'{col}_').columns.to_list() for col in one_hot_encoded_columns}
        data_dict['cyclic_encoded_columns'] = cyclic_encoded_columns

        # Track data types
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}
        
        # Tracking details of label encoding, one-hot encoding, and cyclic encoding
        label_encoded_dict = {}
        one_hot_encoded_dict = {}
        cyclic_encoded_dict = {}

        # Tracking unique values for label encoded columns
        for col in label_encoded_columns:
            label_encoded_dict[col] = df[col].unique().tolist()

        # Tracking one-hot encoded columns and their resulting dummy variables
        for col in one_hot_encoded_columns:
            one_hot_encoded_dict[col] = df.filter(like=f'{col}_').columns.to_list()

        # Tracking original values and transformed values for cyclically encoded columns
        for col in cyclic_encoded_columns:
            cyclic_encoded_dict[col] = {
                'original_values': df[col].unique().tolist(),
                'transformed_columns': [f'{col}_sin', f'{col}_cos']
            }

        # Adding these details to the data dictionary
        data_dict['label_encoded_columns'] = label_encoded_dict
        data_dict['one_hot_encoded_columns'] = one_hot_encoded_dict
        data_dict['cyclic_encoded_columns'] = cyclic_encoded_dict

        # Track the date and time the data was processed
        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
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
        mlflow.start_run(run_name='Feature_Engineering_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
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
    
    @task()
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
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
