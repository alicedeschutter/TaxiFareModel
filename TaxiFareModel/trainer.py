from TaxiFareModel.pipeline import TaxifarePipeline
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from termcolor import colored
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from google.cloud import storage

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[SE] [Stockholm] [alicedeschutter] linear + 1"
BUCKET_NAME="wagon-data-701-deschutter"


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipeline = TaxifarePipeline()
        self.pipeline = pipeline.create_pipeline()

    def run(self):
        """set and train the pipeline"""
        self.trained_model = self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        RMSE = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('rmse', RMSE)
        return round(RMSE, 2)

    def gridsearch(self):
        self.set_pipeline()
        params = {'model':[LinearRegression(), Lasso(), Ridge(), RandomForestRegressor()]}
        gridsearch = GridSearchCV(self.pipeline, params)
        gridsearch.fit(self.X, self.y)
        self.pipeline = gridsearch.best_estimator_
        print(f"Grid Search Performed: {gridsearch.best_params_['model']} scoring is {gridsearch.best_score_}")

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """

        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    def upload_model(self, bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"
        # The path to your file to upload
        # source_file_name = "local/path/to/file"
        # The ID of your GCS object
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )


if __name__ == "__main__":

    # get data
    df = get_data(nrows=1000)

    # clean data
    clean_df = clean_data(df)

    # set X and y
    y = clean_df["fare_amount"]
    X = clean_df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # train
    trainer = Trainer(X_train, y_train)

    trainer.set_pipeline()
    # evaluate

    trainer.gridsearch()

    trainer.run()
    print(trainer.evaluate(X_val, y_val))
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

    #crossvalidate()

    trainer.save_model()
    trainer.upload_model(BUCKET_NAME, "model.joblib", "models/model.joblib")
