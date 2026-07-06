import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def log_training_results(args):
    # Load dataset
    print("Loading preprocessed dataset...")
    df = pd.read_csv("titanic_preprocessed.csv")
    
    # Split features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Log training parameters
    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf
    })
    
    # Initialize and train model
    print("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print("Model training completed successfully.")
    print(f"Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-split", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()

    # Check if run by MLflow Project runner
    if "MLFLOW_RUN_ID" in os.environ:
        print("Running inside an active MLflow Project run environment.")
        log_training_results(args)
    else:
        print("Running manually outside MLflow Project environment.")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Titanic_MLProject_Workflow")
        with mlflow.start_run():
            log_training_results(args)

if __name__ == "__main__":
    main()
