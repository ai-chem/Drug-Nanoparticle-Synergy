import pandas as pd
from catboost import CatBoostRegressor
import joblib
import warnings

warnings.filterwarnings('ignore')

# Model paths
Cat_model_np_path = r'C:\Users\user\Desktop\Synergy_project_2024\catboost_model_mic_np.pkl'
Cat_model_drug_path = r'C:\Users\user\Desktop\Synergy_project_2024\xgb_model_mic_drug.pkl'
Cat_model_drug_np_path = r'C:\Users\user\Desktop\Synergy_project_2024\catboost_model_mic_drug_np.pkl'

def model_load(path):
    return joblib.load(path)

# Load models
mod_np = model_load(Cat_model_np_path)
mod_drug = model_load(Cat_model_drug_path)
mod_drug_np = model_load(Cat_model_drug_np_path)

def np_predict(input_data):
    """Predict using the NP model."""
    x = pd.DataFrame(input_data)
    prediction = mod_np.predict(x)
    return prediction

def drug_predict(input_data):
    """Predict using the Drug model."""
    x = pd.DataFrame(input_data)
    prediction = mod_drug.predict(x)
    return prediction

def drug_np_predict(input_data):
    """Predict using the Drug_NP model."""
    x = pd.DataFrame(input_data)
    mod_drug_np
    prediction = mod_drug_np.predict(x)
    return prediction
