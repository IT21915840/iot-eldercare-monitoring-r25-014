import joblib
import pandas as pd
import warnings
from sklearn.base import InconsistentVersionWarning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning,
    module="sklearn"
)

loaded_model = joblib.load('Sleep/sleep_stage_model.joblib')
loaded_scaler = joblib.load('Sleep/scaler.joblib')


def predict_sleep_stage(spo2: float, hr: float, temp: float) -> str:
    feature_names = ['spo2', 'hr', 'temp']
    sample_data = pd.DataFrame([[spo2, hr, temp]], columns=feature_names)

    sample_data_scaled_array = loaded_scaler.transform(sample_data)

    sample_data_scaled = pd.DataFrame(
        sample_data_scaled_array,
        columns=feature_names
    )

    predicted_sleep_stage = loaded_model.predict(sample_data_scaled)
    return predicted_sleep_stage[0]


if __name__ == "__main__":
    stage = predict_sleep_stage(96, 65, 36.5)
    print("Predicted Stage:", stage)