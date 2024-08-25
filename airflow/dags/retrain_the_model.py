from datetime import timedelta

from airflow.decorators import dag, task

MARKDOWN_TEXT = """
# Retrain the model

This DAG retrain the model based on new data, tests the previous model, and put in production the new model 
if it performs better than the previous one.
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
    dag_id='retrain_the_model',
    description='Retrain the model based on new data, tests the previous model, and put in production the new model if it performs better than the previous one.',
    doc_md=MARKDOWN_TEXT,
    tags=['retrain', 'Bike Sharing Demand'],
    default_args=default_args,
    catchup=False,
)
def retrain_the_model():
    @task.virtualenv(
        task_id='train_challenger_model',
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def train_challenger_model():
        import datetime
        import mlflow
        import awswrangler as wr
        
        from sklearn.base import clone
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from mlflow.models import infer_signature
        
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        def load_champion_model() -> None:
            model_name = 'bike_sharing_model_prod'
            alias = 'champion'
            
            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(name=model_name, alias=alias)
            
            champion_version = mlflow.sklearn.load_model(model_data.source)
            
            return champion_version
        
        def load_the_train_test_data() -> tuple:
            X_train = wr.s3.read_csv("s3://data/train/bike_sharing_demand_X_train_scaled.csv").values
            y_train = wr.s3.read_csv("s3://data/train/bike_sharing_demand_y_train.csv").values.ravel()
            X_test = wr.s3.read_csv("s3://data/test/bike_sharing_demand_X_test_scaled.csv").values
            y_test = wr.s3.read_csv("s3://data/test/bike_sharing_demand_y_test.csv").values.ravel()
            
            return X_train, y_train, X_test, y_test
        
        def mlflow_track_experiment(model, X):
            
            experiment = mlflow.set_experiment("Bike Sharing Demand")
            
            mlflow.start_run(
                run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                experiment_id=experiment.experiment_id,
                tags={"experiment": "challenger models", "dataset": "bike sharing demand"},
                log_system_metrics=True,
            )
    
            params = model.get_params()
            
            params["model"] = type(model).__name__
            
            mlflow.log_params(params)
            
            artifact_path = "model"
            
            signature = infer_signature(X, model.predict(X))
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format='cloudpickle',
                registered_model_name='bike_sharing_model_dev',
                metadata={"model_data_version": 1}
            )
            
            return mlflow.get_artifact_uri(artifact_path)
    
        def register_challenger(model, r2, rmse, model_uri):
            client = mlflow.MlflowClient()
            name = "bike_sharing_model_prod"
            
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["r2"] = r2
            tags["rmse"] = rmse
            
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )
            
            client.set_registered_model_alias(name, "challenger", result.version)
            
        champion_model = load_champion_model()
        
        challenger_model = clone(champion_model)
        
        X_train, y_train, X_test, y_test = load_the_train_test_data()
        
        challenger_model.fit(X_train, y_train)
        
        y_pred = challenger_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        
        artifact_uri = mlflow_track_experiment(challenger_model, X_train)
        
        register_challenger(challenger_model, r2, rmse, artifact_uri)
    
    @task.virtualenv(
        task_id='evaluate_champion_challenge',
        requirements=["scikit-learn==1.3.2",
                        "mlflow==2.10.2",
                        "awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def evaluate_champion_challenge():
        import mlflow
        import awswrangler as wr
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        def load_model(alias):
            model_name = 'bike_sharing_model_prod'
            
            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)
            
            model = mlflow.sklearn.load_model(model_data.source)
            
            return model
        
        def load_the_test_data() -> tuple:
            X_test = wr.s3.read_csv("s3://data/test/bike_sharing_demand_X_test_scaled.csv").values
            y_test = wr.s3.read_csv("s3://data/test/bike_sharing_demand_y_test.csv").values.ravel()
            
            return X_test, y_test
        
        def promote_challenger(name):
            client = mlflow.MlflowClient()
            
            client.delete_registered_model(name, "champion")
            
            challenger_version = client.get_model_version_by_alias(name, "challenger")
            
            client.delete_registered_model_alias(name, "challenger")

            client.set_registered_model_alias(name, "champion", challenger_version.version)
            
        def demote_challenger(name):

            client = mlflow.MlflowClient()

            client.delete_registered_model_alias(name, "challenger")
            
        champion_model = load_model("champion")
        
        challenger_model = load_model("challenger")
        
        X_test, y_test = load_the_test_data()
        
        champion_y_pred = champion_model.predict(X_test)
        challenger_y_pred = challenger_model.predict(X_test)
        
        champion_r2 = r2_score(y_test, champion_y_pred)
        challenger_r2 = r2_score(y_test, challenger_y_pred)
        
        champion_rmse = mean_squared_error(y_test, champion_y_pred)
        challenger_rmse = mean_squared_error(y_test, challenger_y_pred)
        
        experiment = mlflow.set_experiment("Bike Sharing Demand")
        
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")
        
        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_r2_champion", champion_r2)
            mlflow.log_metric("test_r2_challenger", challenger_r2)
            
            mlflow.log_metric("test_rmse_champion", champion_rmse)
            mlflow.log_metric("test_rmse_challenger", challenger_rmse)
            
    
        name = "bike_sharing_model_prod"
        
        if challenger_r2 > champion_r2 and challenger_rmse < champion_rmse:
            print("Challenger model is better than the champion model")
            promote_challenger(name)
        else:
            print("Challenger model is not better than the champion model")
            demote_challenger(name)
            
    train_challenger_model() >> evaluate_champion_challenge()
    
retrain_the_model_dag = retrain_the_model()
            