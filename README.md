# TP Final Aprendizaje de Máquina II - Bike Sharing Demand

Este repositorio contiene el trabajo práctico final de la materia Aprendizaje de Máquina II de la Especialización en Inteligencia Artificial (CEIA) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA). El objetivo es implementar un modelo productivo para predecir la demanda de bicicletas dado un conjunto de datos de datos históricos. El dataset utilizado es el [Bike Sharing Demand - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

#### El video demostrativo del funcionaiento del proyecto se encuentra en el siguiente [link](https://youtu.be/mq8LHZM73UM?si=BUXOC1NrGS7a6jWh)

Nuestra implementación incluye:

- En **Apache Airflow**, un DAG que orquesta el flujo de extracción, transformación y carga de los datos en un bucket de S3 (Minio) `s3://data`. También se utiliza **MLFlow** para el seguimiento de estos procesos.

- Dos **Jupyter notebooks** que contienen la exploración de los datos y el entrenamiento de los modelos. El primero se encuentra en [data_exploration.ipynb](notebooks/data_exploration.ipynb) y el segundo en [training_experiments.ipynb](notebooks/training_experiments.ipynb). En este último se realiza una búsqueda de hiperparámetros para encontrar el mejor modelo de regresión. Toda la experimentación se registra mediante **MLFlow**, se generan gráficos de importancia de características, correlaciones y se registra el mejor modelo entrenado con sus correspondientes métricas, parámetros y el alias de `champion`.

- Un servidor **FastAPI** que expone un endpoint `/predict` para realizar predicciones de la demanda de bicicletas. Además, un endpoint `/reload_model` para poder recargar el modelo que se requiera, dado un nombre y un alias. El código principal se encuentra en [app.py](dockerfiles/fastapi/app.py). También se incluye una interfaz web sencilla para realizar predicciones en [Bike Sharing Demand](http://localhost:8800).

- En **Apache Airflow**, un DAG que orquesta un flujo de reentrenamiento del modelo. Se compara el nuevo modelo `challenger` con el `champion` y si el primero es mejor, se actualiza a `champion`. El proceso se registra en **MLFlow**.

## Instrucciones para levantar el proyecto
1. En un primer punto es necesario asegurarse de tener instalado [Docker](https://docs.docker.com/get-docker/) y [Docker Compose](https://docs.docker.com/compose/install/). 

2. Clonar el repositorio y moverse al directorio raíz del proyecto.

3. Crea las carpetas `airflow/config`, `airflow/dags`, `airflow/logs`, `airflow/plugins`, `airflow/logs`.
```bash
mkdir -p airflow/config airflow/dags airflow/logs airflow/plugins airflow/logs
```

4. Crear un archivo `.env` en la raíz del proyecto, donde se definirán las variables de entorno necesarias para el proyecto. En el archivo `.env.template` se encuentran variables de ejemplo que se pueden utilizar.

5. En caso de estar utilizando Linux o MacOS, en el archivo `.env`, se debe reemplazar la variable `AIRFLOW_UID` por el correspondiente al usuario que esté utilizando el sistema operativo. Para obtener el UID del usuario, se puede ejecutar el comando `id -u <username>`. 

6. En la carpeta raíz del proyecto, ejecutar el siguiente comando para levantar el multi-contenedor:
```bash
docker compose --profile all up
```

7. Una vez que todos los contenedores estén levantados, se puede acceder a los diferentes servicios:
- **Apache Airflow**: [http://localhost:8080](http://localhost:8080)
- **MLFlow**: [http://localhost:5000](http://localhost:5000)
- **Minio**: [http://localhost:9000](http://localhost:9001)
- **FastAPI**: [http://localhost:8800](http://localhost:8800)
- **FastAPI Docs**: [http://localhost:8800/docs](http://localhost:8800/docs)

## Detener los contenedores

Para detener los contenedores, ejecutar el siguiente comando:
```bash
docker compose --profile all down
```

Si se desea eliminar los volúmenes creados por los contenedores, se puede ejecutar el siguiente comando:
```bash
docker compose --profile all down --volumes
```

## Instrucciones para probar el funcionamiento del proyecto

1. Una vez levantado el multi-contenedor, ejecutar en Airflow el DAG `process_etl_bike_sharing_data`, de esta manera se crearán los datos en el bucket `s3://data`.

2. Ejecutar el notebook [data_exploration.ipynb](notebooks/data_exploration.ipynb) para realizar la búsqueda de 
hiperparámetros y entrenar el mejor modelo.

3. Utilizar el servicio de API corriendo en [Bike Sharing Demand](http://localhost:8800), también se puede acceder a la documentación de la API en [Bike Sharing Demand - Docs](http://localhost:8800/docs).

4. Ejecutar el DAG `retrain_the_model` en Apache Airflow para reentrenar el modelo y comparar si un nuevo modelo entrenado es mejor que el actual. Si es así, se actualizará el modelo `champion`. Es importante mencionar que se debe volver a correr el DAG `process_etl_bike_sharing_data` para volver a obtener los datos de entrenamiento, antes de ejecutar el DAG `retrain_the_model`, para que el nuevo modelo se entrene con nuevos datos y tenga posibilidad de sobrepasar al modelo `champion` anterior.

## Integrantes
- **Christian Ricardo Conchari Cabrera** - chrisconchari@gmail.com
- **William Andrés Prada Mancilla** - wpradamancilla@gmail.com
- **Carlos Villalobos** - carlosvillalobosh3@gmail.com
- **Carlos Mendez** - carlos.mendezt@gmail.com
- **German Poletto** - germanpp13@gmail.com
