from TaxiFareModel.pipeline import TaxifarePipeline
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipeline = TaxifarePipeline()
        self.pipe = pipeline.create_pipeline()

    def run(self):
        """set and train the pipeline"""
        self.pipe.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        RMSE = compute_rmse(y_pred, y_test)
        return RMSE

if __name__ == "__main__":

    # get data
    df = get_data()

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
    trainer.run()
    # evaluate

    print(trainer.evaluate(X_val, y_val))
