from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib

app = FastAPI()

# if add_middleware is not used, the API will only work when called from Back-End code (or similar)
# now --> allows it to be called from the JavaScript code of a web page.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, passenger_count):

    # creation of a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, '%Y-%m-%d %H:%M:%S')

    # localization of the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    # localize the datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    formatted_pickup_datetime = utc_pickup_datetime.strftime((
        "%Y-%m-%d %H:%M:%S UTC"))

    X_pred = pd.DataFrame({
        'key':
        ['2013-07-06 17:18:00.000000119'],  #hard coded value to match X_train format
        'pickup_datetime': [formatted_pickup_datetime],
        'pickup_longitude': [float(pickup_longitude)],
        "pickup_latitude": [float(pickup_latitude)],
        "dropoff_longitude": [float(dropoff_longitude)],
        "dropoff_latitude": [float(dropoff_latitude)],
        "passenger_count": [int(passenger_count)]
    })

    loaded_model = joblib.load('model.joblib')
    y_pred = loaded_model.predict(X_pred)

    return {"prediction": float(y_pred)}
