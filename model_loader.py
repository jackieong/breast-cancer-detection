import pickle
import joblib

def load_model(storage_method: str):
    if storage_method == "pickle":
        # Load the model
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
    elif storage_method == "joblib":
        model = joblib.load('model.joblib')

    return model

def save_model(model, storage_method: str):
    if storage_method == "pickle":
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
    elif storage_method == "joblib":
        joblib.dump(model, 'model.joblib')
