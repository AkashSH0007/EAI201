import pandas as pd
import joblib 
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle


print("--- Loading Final Dataset ---")
try:
    final_df = pd.read_csv('world_cup_final_dataset.csv')
    print("SUCCESS: Final dataset loaded!")
except FileNotFoundError:
    print("ERROR: 'world_cup_final_dataset.csv' not found. Make sure it's in the same folder.")
    exit()

# Task 2: Model Building and Training 
print("\n--- Task 2: Preparing Data for Modeling ---")

#  Step 1: Feature Selection 
target = 'match_outcome'
features = [
    'neutral',
    'home_team_avg_age', 'home_team_total_market_value', 'home_team_avg_experience',
    'away_team_avg_age', 'away_team_total_market_value', 'away_team_avg_experience',
    'home_team_rank', 'home_team_rank_points',
    'away_team_rank', 'away_team_rank_points',
    'home_team_win_rate', 'away_team_win_rate'
]
print(f"Features selected: {len(features)}")
print(f"Target selected: {target}")

# --- Create X and y ---
try:
    X = final_df[features]
    y = final_df[target]
    
    X.loc[:, 'neutral'] = X['neutral'].astype(int)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
except KeyError as e:
    print(f"\nERROR: A feature column name might be wrong: {e}")
    exit()
except Exception as e:
    print(f"\nAn error occurred preparing X and y: {e}")
    exit()

# --- Step 2: Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# --- Step 3: Preprocessing - Scale Numerical Features ---
scaler = StandardScaler()
scaler.fit(X_train) 
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Numerical features scaled.")

# --- Step 4: Train Logistic Regression ---
print("\n--- Training Logistic Regression Model ---")

log_reg_model = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
log_reg_model.fit(X_train_scaled, y_train)
print("SUCCESS: Logistic Regression model trained.")

# --- Step 5: Train Random Forest ---
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
print("SUCCESS: Random Forest model trained.")



# --- Task 3: Model Evaluation ---
print("\n--- Task 3: Model Evaluation on Test Data ---")
y_pred_lr = log_reg_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)
print("Predictions complete.")

# 2. Calculate Metrics & Reports 
print("\n--- Evaluating Logistic Regression Model ---")
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {accuracy_lr:.4f}")
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr, zero_division=0))

print("\n--- Evaluating Random Forest Model ---")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf:.4f}")
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# --- 3. Generate Visualizations (Confusion Matrices) ---
print("\n--- Generating Confusion Matrices ---")
try:
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Draw', 'HomeWin', 'AwayWin'], 
                yticklabels=['Draw', 'HomeWin', 'AwayWin'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.savefig('confusion_matrix_lr.png')
    plt.close()
    print("-> Confusion Matrix saved as 'confusion_matrix_lr.png'")
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Draw', 'HomeWin', 'AwayWin'],
                yticklabels=['Draw', 'HomeWin', 'AwayWin'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Random Forest')
    plt.savefig('confusion_matrix_rf.png')
    plt.close()
    print("-> Confusion Matrix saved as 'confusion_matrix_rf.png'")
except Exception as e:
    print(f"ERROR generating Confusion Matrix: {e}")

# --- 4. Generate Visualizations (ROC Curves) ---
print("\n--- Generating ROC Curves & AUC ---")
try:
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_binarized.shape[1]
    
    # LR ROC
    y_score_lr = log_reg_model.predict_proba(X_test_scaled)
    fpr_lr, tpr_lr, roc_auc_lr = dict(), dict(), dict()
    for i in range(n_classes):
        fpr_lr[i], tpr_lr[i], _ = roc_curve(y_test_binarized[:, i], y_score_lr[:, i])
        roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])
    roc_auc_macro_lr = roc_auc_score(y_test_binarized, y_score_lr, average='macro')
    print(f"-> Logistic Regression Macro-Average ROC AUC: {roc_auc_macro_lr:.4f}")
    
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr_lr[i], tpr_lr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc_lr[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('Multi-class ROC Curve - Logistic Regression (OvR)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_lr.png')
    plt.close()
    print("-> ROC Curve saved as 'roc_curve_lr.png'")

    # RF ROC
    y_score_rf = rf_model.predict_proba(X_test_scaled)
    fpr_rf, tpr_rf, roc_auc_rf = dict(), dict(), dict()
    for i in range(n_classes):
        fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_binarized[:, i], y_score_rf[:, i])
        roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])
    roc_auc_macro_rf = roc_auc_score(y_test_binarized, y_score_rf, average='macro')
    print(f"-> Random Forest Macro-Average ROC AUC: {roc_auc_macro_rf:.4f}")

    plt.figure(figsize=(8, 6))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr_rf[i], tpr_rf[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc_rf[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('Multi-class ROC Curve - Random Forest (OvR)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_rf.png')
    plt.close()
    print("-> ROC Curve saved as 'roc_curve_rf.png'")
except Exception as e:
    print(f"ERROR generating ROC Curves: {e}")
print("\n--- Task 3 Complete ---")


# --- Task 4: Feature Importance ---
print("\n--- Task 4: Feature Importance ---")
try:
    # Logistic Regression Coefficients
    feature_names = X.columns
    
    avg_lr_importance = np.mean(np.abs(log_reg_model.coef_), axis=0)
    lr_feature_importance = pd.DataFrame({'feature': feature_names, 'importance': avg_lr_importance})
    lr_feature_importance = lr_feature_importance.sort_values(by='importance', ascending=False)
    
    print("\nLogistic Regression Feature Importance (Mean Abs Coefficient):")
    print(lr_feature_importance.head(10))

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=lr_feature_importance.head(15))
    plt.title('Top 15 Feature Importances (Logistic Regression)')
    plt.tight_layout()
    plt.savefig('feature_importance_lr.png')
    plt.close()
    print("-> LR Feature Importance plot saved as 'feature_importance_lr.png'")

    # Random Forest Feature Importances
    rf_importance = rf_model.feature_importances_
    rf_feature_importance = pd.DataFrame({'feature': feature_names, 'importance': rf_importance})
    rf_feature_importance = rf_feature_importance.sort_values(by='importance', ascending=False)
    
    print("\nRandom Forest Feature Importance:")
    print(rf_feature_importance.head(10))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=rf_feature_importance.head(15))
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.close()
    print("-> RF Feature Importance plot saved as 'feature_importance_rf.png'")
except Exception as e:
    print(f"ERROR in Feature Importance: {e}")
print("\n--- Task 4 Complete ---")


print("\n--- Saving Models and Scaler for Prediction ---")
try:
    # Save the Logistic Regression model (our best model)
    joblib.dump(log_reg_model, 'logistic_regression_model.joblib')
    print("SUCCESS: Logistic Regression model saved as 'logistic_regression_model.joblib'")
    
    # Save the Random Forest model
    joblib.dump(rf_model, 'random_forest_model.joblib')
    print("SUCCESS: Random Forest model saved as 'random_forest_model.joblib'")
    
    # Save the StandardScaler
    joblib.dump(scaler, 'scaler.joblib')
    print("SUCCESS: Scaler saved as 'scaler.joblib'")
    
    print("\nAll models and scaler are saved. You can now run predict.py.")

except Exception as e:
    print(f"ERROR: Could not save models. {e}")

print("\n--- All Tasks Complete ---")

