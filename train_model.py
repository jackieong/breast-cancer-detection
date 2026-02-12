# train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from model_pickle import save_model_pickle, load_model_pickle



data = load_breast_cancer()  # load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)
# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Save the model - Pickle
save_model_pickle(model)

print("Model trained and saved as model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print(f"Input length: {len(X_test)} \nPrediction: {len(y_pred)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

