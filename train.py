import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import os
import seaborn as sns
import joblib
import numpy as np

def main():
    # Download penguins dataset
    print("Loading data...")
    df = sns.load_dataset("penguins")
    print(f"Initial dataset shape: {df.shape}")
    
    # Drop rows with missing values
    df = df.dropna()
    print(f"Dataset shape after dropping missing values: {df.shape}")
    
    # Encode target variable
    print("\nEncoding target variable...")
    le = LabelEncoder()
    y = le.fit_transform(df["species"])
    
    # One-hot encode categorical features
    print("Encoding categorical features...")
    X = pd.get_dummies(df.drop("species", axis=1), columns=["sex", "island"])
    
    # Ensure consistent column order
    expected_columns = [
        'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
        'year', 'sex_Female', 'sex_Male', 'island_Biscoe', 'island_Dream',
        'island_Torgensen'
    ]
    X = X.reindex(columns=expected_columns, fill_value=0)
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    print("\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Train model
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='multi:softprob',
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    print("\nModel training complete.")
    
    # Print class info
    n_classes = len(le.classes_)
    print(f"\nNumber of classes: {n_classes}")
    print(f"Class names: {list(le.classes_)}")
    
    # Evaluate
    print("\nEvaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print("\nTrain classification report:")
    print(classification_report(y_train, train_pred, target_names=le.classes_))
    
    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, target_names=le.classes_))
    
    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_pred))
    
    # Save model and artifacts
    print("\nSaving model and artifacts...")
    os.makedirs("app/data", exist_ok=True)
    
    # Save model
    model.save_model("app/data/model.json")
    
    # Save the label encoder
    joblib.dump(le, "app/data/label_encoder.joblib")
    
    # Save the feature names
    with open("app/data/feature_names.txt", "w") as f:
        f.write("\n".join(X.columns))
    
    print("\nModel and artifacts saved to app/data/")
    
    return model, le

if __name__ == "__main__":
    model, le = main()
