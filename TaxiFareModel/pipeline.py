from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer

class TaxifarePipeline:

    def __init__(self):
        pass

    def create_pipeline(self):

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())])
        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        from sklearn.linear_model import LinearRegression

        # Add the model of your choice to the pipeline
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        return pipe
