import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define models for set 2
MODEL_SET2 = {
    'SVM': SVC(probability=True, random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

def generate_hybrid_data_set2(data_path, output_path):
    """
    Generate augmented dataset with binary predictions from Set 2 models:
    SVM, ExtraTrees, LogisticRegression
    
    Args:
        data_path (str): Path to the original CSV data file
        output_path (str): Path to save the augmented dataset
    """
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # Last column is the target 'defects'
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Make copies of original train and test data
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()
    
    # Scale features for training the models
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names for the scaled data
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Train models and add binary predictions to the original data
    print("Training Set 2 models and generating predictions...")
    for name, model in MODEL_SET2.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Generate predictions on scaled data
        train_proba = model.predict_proba(X_train_scaled)[:, 1]
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Convert to binary predictions (0 or 1)
        train_preds = (train_proba >= 0.5).astype(int)
        test_preds = (test_proba >= 0.5).astype(int)
        
        # Add binary predictions as new features to the ORIGINAL data
        X_train_original[f'{name}_pred'] = train_preds
        X_test_original[f'{name}_pred'] = test_preds
        
        # Evaluate model
        accuracy = accuracy_score(y_test, test_preds)
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Add target column back to the original data with predictions
    X_train_augmented = X_train_original.copy()
    X_test_augmented = X_test_original.copy()
    
    X_train_augmented['defects'] = y_train.values
    X_test_augmented['defects'] = y_test.values
    
    # Combine train and test for final dataset
    augmented_data = pd.concat([X_train_augmented, X_test_augmented])
    
    # Save augmented dataset
    augmented_data.to_csv(output_path, index=False)
    print(f"Augmented dataset with Set 2 binary predictions saved to {output_path}")
    print(f"New dataset shape: {augmented_data.shape}")
    
    return augmented_data

if __name__ == "__main__":
    # Example usage
    data_path = "data/pc1_full.csv"
    output_path = "data/set2/pc1_fulls2.csv"
    augmented_data = generate_hybrid_data_set2(data_path, output_path)