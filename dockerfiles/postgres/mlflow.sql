-- Check if database exists, and create it if it doesn't
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db') THEN
      CREATE DATABASE mlflow_db;
   END IF;
END
$$;