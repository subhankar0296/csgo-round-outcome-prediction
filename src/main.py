"""
Counter-Strike Competitive Match Round Outcome Prediction Pipeline
Author: Subhankar
Description: End-to-end data preparation, dimensionality reduction, 
             ensemble classification, and visual graphics diagnostics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_and_clean_data(file_path):
    print("--- Ingesting Snapshot Raw Dataset ---")
    df = pd.read_csv(file_path, low_memory=False)
    df.dropna(inplace=True)
    return df


def preprocess_features(df, plot_dir):
    print("--- Executing Feature Transformation & Targeted Plotting ---")
    
    # 1. GENERATE VISUALIZATION: Target Variable Count Plot
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    sns.countplot(x='round_winner', data=df, palette='Set2')
    plt.title('Class Balance Analysis: Round Winner Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Winning Faction Faction')
    plt.ylabel('Total Round Record Snapshots')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '01_class_balance_countplot.png'), dpi=300)
    plt.close()
    print(" -> Success: Saved '01_class_balance_countplot.png'")
    
    # Process categorical transformations
    le_map = LabelEncoder()
    le_bomb = LabelEncoder()
    le_winner = LabelEncoder()
    
    df['map'] = le_map.fit_transform(df['map'])
    df['bomb_planted'] = le_bomb.fit_transform(df['bomb_planted'])
    df['round_winner'] = le_winner.fit_transform(df['round_winner'])
    
    X = df.drop(columns=['round_winner'])
    y = df['round_winner']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def run_random_forest_pipeline(X_train, X_test, y_train, y_test, feature_names, plot_dir):
    print("\n--- Training Fine-Tuned Random Forest Ensemble Model ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Ensemble Accuracy: {rf_acc:.4f}")
    
    # 2. GENERATE VISUALIZATION: Top Feature Importances Plot
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Target Top 15 variables
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices], palette='viridis', hue=feature_names[indices], legend=False)
    plt.title('Feature Performance Weight Matrix (Top 15 Predictors)', fontsize=12, fontweight='bold')
    plt.xlabel('Relative Feature Importance Score')
    plt.ylabel('Predictor Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02_feature_importance_matrix.png'), dpi=300)
    plt.close()
    print(" -> Success: Saved '02_feature_importance_matrix.png'")
    
    # 3. GENERATE VISUALIZATION: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['CT Win', 'T Win'], yticklabels=['CT Win', 'T Win'],
                annot_kws={"size": 12, "weight": "bold"})
    plt.title('Ensemble Model Performance: Confusion Matrix Heatmap', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label Outcomes')
    plt.ylabel('Actual Truth Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '03_model_confusion_matrix.png'), dpi=300)
    plt.close()
    print(" -> Success: Saved '03_model_confusion_matrix.png'")
    
    print("\nDetailed Performance Matrix:")
    print(classification_report(y_test, y_pred_rf, target_names=['CT Win', 'T Win']))
    return rf


if __name__ == "__main__":
    DATA_PATH = "D:\Data Analytics Project\csgo_round_snapshots.csv"
    PLOT_OUTPUT_DIR = "plots"
    
    # Establish local folder directories safety parameters
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    try:
        raw_df = load_and_clean_data(DATA_PATH)
        X_train, X_test, y_train, y_test, feat_cols = preprocess_features(raw_df, PLOT_OUTPUT_DIR)
        rf_model = run_random_forest_pipeline(X_train, X_test, y_train, y_test, feat_cols, PLOT_OUTPUT_DIR)
        print("\n--- Diagnostic Pipeline Complete with Saved Visualization Assets ---")
        
    except FileNotFoundError:
        print(f"Error: Please save your raw dataset inside '{DATA_PATH}' location context.")