import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_improved_nn(input_dim, class_weights=None):
    """
    Build an improved neural network with better architecture for software defect prediction
    
    Args:
        input_dim (int): Input dimension (number of features)
        class_weights (dict): Optional class weights for imbalanced data
        
    Returns:
        model: Compiled TensorFlow model
    """
    model = Sequential([
        # Input layer with L1L2 regularization
        Dense(128, activation='relu', input_dim=input_dim, 
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        # First hidden layer
        Dense(64, activation='relu', 
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second hidden layer
        Dense(32, activation='relu', 
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model with RMSprop optimizer as mentioned in the paper
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def preprocess_data(df, target_col='defects', apply_smote=True):
    """
    Comprehensive data preprocessing including scaling and handling class imbalance
    
    Args:
        df (DataFrame): Input dataframe
        target_col (str): Target column name
        apply_smote (bool): Whether to apply SMOTE for handling class imbalance
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    print("\nData Preprocessing:")
    print(f"Original data shape: {df.shape}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found: {missing[missing > 0]}")
        # Fill missing values with median
        df = df.fillna(df.median())
    else:
        print("No missing values found")
    
    # Check for class imbalance
    y = df[target_col]
    class_counts = y.value_counts()
    print(f"Class distribution:\n{class_counts}")
    imbalance_ratio = class_counts.min() / class_counts.max()
    print(f"Class imbalance ratio: {imbalance_ratio:.4f}")
    
    # Prepare features and target
    X = df.drop(target_col, axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=59, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to address class imbalance
    if apply_smote and imbalance_ratio < 0.5:
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE - training data shape: {X_train_scaled.shape}")
        print(f"After SMOTE - class distribution: {pd.Series(y_train).value_counts()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_improved_nn(data_path, model_save_path=None, apply_smote=True, scale_data=True):
    """
    Train an improved neural network with better techniques for higher accuracy
    
    Args:
        data_path (str): Path to the augmented dataset CSV
        model_save_path (str): Path to save the trained model
        apply_smote (bool): Whether to apply SMOTE for class imbalance
        scale_data (bool): Whether to scale the data
        
    Returns:
        dict: Results including model, metrics, and history
    """
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Option 1: Use preprocessing function
    if scale_data:
        X_train, X_test, y_train, y_test, _ = preprocess_data(df, apply_smote=apply_smote)
    else:
        # Option 2: Skip scaling but still split data
        X = df.drop('defects', axis=1)
        y = df['defects']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=59, stratify=y
        )
        print(f"Training data shape (no scaling): {X_train.shape}")
    
    # Calculate class weights if not using SMOTE
    class_weights = None
    if not apply_smote:
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        print(f"Class weights: {class_weights}")
    
    # Create directories for model and plots
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Build the improved neural network
    input_dim = X_train.shape[1]
    model = build_improved_nn(input_dim, class_weights)
    model.summary()
    
    # Callbacks for training
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Add model checkpoint if save path provided
    if model_save_path:
        callbacks.append(
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    # Train the model
    print("\nTraining improved neural network...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate all metrics mentioned in the paper
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Print metrics
    print("\nPerformance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize training history
    if model_save_path:
        plot_dir = os.path.dirname(model_save_path)
        
        # Learning curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/learning_curves.png")
        
        # Confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/confusion_matrix.png")
        
        print(f"\nTraining plots saved to {plot_dir}")
    
    return {
        'model': model,
        'metrics': metrics,
        'history': history,
        'predictions': y_pred,
        'y_test': y_test
    }

def train_all_models(scale_data=True):
    """
    Train and evaluate all three hybrid models
    
    Args:
        scale_data (bool): Whether to scale the data
        
    Returns:
        dict: Results for all models
    """
    # Define paths
    data_paths = {
        "set1": "data/set1/jm1_fulls1.csv",
        "set2": "data/set2/jm1_fulls2.csv",
        "set3": "data/set3/jm1_fulls3.csv"
    }
    
    results = {}
    
    # Create output directory
    os.makedirs("models/jm1_full_59", exist_ok=True)
    
    # Train each model
    for set_name, path in data_paths.items():
        print("\n" + "="*60)
        print(f"Training improved neural network for {set_name}")
        print("="*60)
        
        model_results = train_improved_nn(
            path, 
            model_save_path=f"models/jm1_full_59/nn_jm1_full_59{set_name}.h5",
            scale_data=scale_data
        )
        
        results[set_name] = model_results['metrics']
    
    # Compare results
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    print(comparison_df)
    
    # Find best model
    best_model = comparison_df['accuracy'].idxmax()
    print(f"\nBest model: {best_model} with {comparison_df.loc[best_model, 'accuracy']} accuracy")
    
    # Save comparison to CSV
    comparison_df.to_csv("models/jm1_full_59/model_comparison_jm1_full_59.csv")
    print("Comparison results saved to models/jm1_full_59/model_comparison_jm1_full_59.csv")
    
    return results

if __name__ == "__main__":
    # Train a single model
    # Uncomment the model you want to train:
    # result = train_improved_nn("data/set1/jm1s1.csv", "models/improved/nn_set1.h5", scale_data=True)
    # result = train_improved_nn("data/set2/jm1s2.csv", "models/improved/nn_set2.h5", scale_data=True)
    # result = train_improved_nn("data/set3/jm1s3.csv", "models/improved/nn_set3.h5", scale_data=True)
    
    # Or train all models
    all_results = train_all_models(scale_data=True)