import pickle

def load_model_pickle():
    # Load the model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def save_model_pickle(model):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)