from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, model_type='rf', dataset_type='tabular'):
    if dataset_type == 'tabular':
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:  # image
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)