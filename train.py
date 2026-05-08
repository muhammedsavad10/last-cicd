import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Starting training with MLflow...")

mlflow.set_tracking_uri("file:./mlruns")
# 🔥 create experiment
mlflow.set_experiment("iris_project")

# load data
iris = pd.read_csv('data/iris.csv')
X = iris.drop(columns=['Species'])
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    # ✅ MLflow logging
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model", "RandomForest")

    # # 🔥 LOG MODEL (VERY IMPORTANT)
    # mlflow.sklearn.log_model(name= "model")

    # save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

print("MLflow logging done ✅")