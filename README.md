# TP Final Aprendizaje de Máquina II - Bike Sharing Demand

Este repositorio contiene el trabajo práctico final de la materia Aprendizaje de Máquina II de la Especialización en Inteligencia Artificial (CEIA) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA). El objetivo es implementar un modelo productivo para predecir la demanda de bicicletas dado un conjunto de datos de datos históricos. El dataset utilizado es el [Bike Sharing Demand - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

Nuestra implementación incluye:

- En **Apache Airflow**, un DAG que orquesta el flujo de extracción, transformación y carga de los datos en un bucket de S3 (Minio) `s3://data`. También se utiliza **MLFlow** para el seguimiento de estos procesos.

- Dos **Jupyter notebooks** que contienen la exploración de los datos y el entrenamiento de los modelos. El primero se encuentra en [data_exploration.ipynb](notebooks/data_exploration.ipynb) y el segundo en [training_experiments.ipynb](notebooks/training_experiments.ipynb). En este último se realiza una búsqueda de hiperparámetros para encontrar el mejor modelo de regresión. Toda la experimentación se registra mediante **MLFlow**, se generan gráficos de importancia de características, correlaciones y se registra el mejor modelo entrenado con sus correspondientes métricas, parámetros y el alias de `champion`.

- Un servidor **FastAPI** que expone un endpoint `/predict` para realizar predicciones de la demanda de bicicletas. Además, un endpoint `/reload_model` para poder recargar el modelo que se requiera, dado un nombre y un alias. El código principal se encuentra en [app.py](src/app.py). También se incluye una interfaz web sencilla para realizar predicciones en [Bike Sharing Demand](https://localhost:8800).

- En **Apache Airflow**, un DAG que orquesta un flujo de reentrenamiento del modelo. Se compara el nuevo modelo `challenger` con el `champion` y si el primero es mejor, se actualiza a `champion`. El proceso se registra en **MLFlow**.

# Instrucciones para probar el funcionamiento del proyecto

1. Una vez levantado el multi-contenedor, ejecutar en Airflow el DAG `process_etl_bike_sharing_data`, de esta manera se crearán los datos en el bucket `s3://data`.

2. Ejecutar el notebook [data_exploration.ipynb](notebooks/data_exploration.ipynb) para realizar la búsqueda de 
hiperparámetros y entrenar el mejor modelo.

3. Utilizar el servicio de API corriendo en [Bike Sharing Demand](https://localhost:8800), también se puede acceder a la documentación de la API en [Bike Sharing Demand - Docs](https://localhost:8800/docs).

4. Ejecutar el DAG `retrain_the_model` en Apache Airflow para reentrenar el modelo y comparar si un nuevo modelo entrenado es mejor que el actual. Si es así, se actualizará el modelo `champion`.

