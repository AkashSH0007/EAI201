import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the file name from Task 1
TASK_1_OUTPUT_FILE = 'task_1_integrated_data.csv'

print(f"--- Task 3: Delta_train_and_evaluate ---")

# Load the dataset
try:
    df = pd.read_csv(TASK_1_OUTPUT_FILE)
    print(f"Successfully loaded '{TASK_1_OUTPUT_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{TASK_1_OUTPUT_FILE}' not found. Cannot proceed with Task 3.")
    # Stop execution if the file isn't found
    raise

# --- A) Prepare data and split ---
print("\n--- A) Preparing and Splitting Data ---")
# Define Features (X)
features = [
    'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator',
    'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail',
    'domestic', 'catsize', 'ecosystem_type', 'predator_score'
]
# Define Target (y)
target = 'Class_Type' # Using the string name for better report/matrix labels

X = df[features]
y = df[target]

# Get sorted list of class names for plots
class_names = sorted(y.unique())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    train_size=0.80, # train_size=0.80 means test_size=0.20
    test_size=0.20,
    random_state=789
)
print(f"Data split: {len(X_train)} train, {len(X_test)} test samples. Random State=789.")


# --- B) Configure and train Random Forest ---
print("\n--- B) Configuring and Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=3,
    random_state=789 # for model reproducibility
)
rf_model.fit(X_train, y_train)
print("Random Forest model trained with n_estimators=200, min_samples_split=3, random_state=789.")


# --- C) Training and Testing Accuracy ---
print("\n--- C) Model Accuracy ---")
# Predictions
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

# Accuracies
train_accuracy = accuracy_score(y_train, y_pred_train_rf)
test_accuracy = accuracy_score(y_test, y_pred_test_rf)

print(f"training accuracy:{train_accuracy:.4f}")
print(f"testing accuracy:{test_accuracy:.4f}")
print(f"overfitting Gap: {train_accuracy - test_accuracy:.4f}")


# --- D) Classification Report ---
print("\n--- D) Classification Report ---")
report_str = classification_report(
    y_test, 
    y_pred_test_rf, 
    labels=class_names,
    zero_division=0
)
print(report_str)

# --- E) Confusion Matrix Heatmap ---
print("\n--- E) Generating Confusion Matrix Heatmap ---")
cm = confusion_matrix(y_test, y_pred_test_rf, labels=class_names)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.title('Random Forest Confusion Matrix (n_est=200, min_split=3, r_state=789)', fontsize=14)
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('Actual Class', fontsize=12)
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png')
print("Saved 'rf_confusion_matrix.png'")
plt.close()


# --- F) Feature Importance Plot ---
print("\n--- F) Generating Feature Importance Plot ---")
importances = rf_model.feature_importances_
feat_imp_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
top_12_features = feat_imp_series.head(12)

# Create color list: red for engineered, steelblue otherwise
engineered_features_names = ['ecosystem_type', 'predator_score']
colors = ['red' if feat in engineered_features_names else 'steelblue' for feat in top_12_features.index]

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x=top_12_features.values, 
    y=top_12_features.index, 
    palette=colors,
    orient='h'
)
ax.set_title('Top 12 Feature Importances (Random Forest)', fontsize=16)
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
# Invert y-axis to show most important at top
ax.invert_yaxis()
# Add a legend for the colors
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Engineered Feature')
blue_patch = mpatches.Patch(color='steelblue', label='Original Feature')
plt.legend(handles=[red_patch, blue_patch])
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
print("Saved 'rf_feature_importance.png'")
plt.close()


# --- G) Train Comparison Model (K-Nearest Neighbors) ---
print("\n--- G) Training Comparison Model (K-NN) ---")
# KNN is sensitive to feature scale, so we scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5) # Default k=5
knn_model.fit(X_train_scaled, y_train)

# Evaluate KNN
knn_accuracy = knn_model.score(X_test_scaled, y_test)
print(f"K-Nearest Neighbors (k=5) model trained and evaluated. Accuracy: {knn_accuracy:.4f}")


# --- H) Critical Analysis Output ---
print("\n--- H) Critical Analysis Output ---")

# 1. Most important feature
top_feature_name = feat_imp_series.index[0]
top_feature_value = feat_imp_series.values[0]

# 2. & 3. Worst and Best performing class
report_dict = classification_report(
    y_test, 
    y_pred_test_rf, 
    labels=class_names,
    output_dict=True,
    zero_division=0
)
class_f1_scores = {}
for class_name in class_names:
    class_f1_scores[class_name] = report_dict[class_name]['f1-score']

# Sort by F1-score
sorted_classes = sorted(class_f1_scores.items(), key=lambda item: item[1])
worst_class_name, worst_class_f1 = sorted_classes[0]
best_class_name, best_class_f1 = sorted_classes[-1]

# 4. Engineered feature rank
# Get full sorted list of feature names
feat_rank_list = feat_imp_series.index.tolist()
# Find the rank of the *highest* ranked engineered feature
your_feature_name = ''
your_feature_rank = -1
# Check in order of rank
for i, feat in enumerate(feat_rank_list):
    if feat in engineered_features_names:
        your_feature_name = feat
        your_feature_rank = i + 1 # +1 for 1-based indexing
        break # Stop at the first (highest) one we find

# 5. Model comparison
rf_acc_for_print = test_accuracy # from step C
knn_acc_for_print = knn_accuracy # from step G

# Print all outputs
print("\nModel Analysis")
print(f"1. Most important feature:{top_feature_name} (importance: {top_feature_value:.3f})")
print(f"2. Worst performing class: {worst_class_name} (F1: {worst_class_f1:.3f})")
print(f"3. Best performing class: {best_class_name} (F1: {best_class_f1:.3f})")
print(f"4. Your engineered feature '{your_feature_name}' ranked #{your_feature_rank}")
print(f"5. Model comparison: KNN={knn_acc_for_print:.3f} vs RF={rf_acc_for_print:.3f}")

print("\n--- Task 3 Complete ---")