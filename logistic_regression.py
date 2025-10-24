# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target (0=Setosa, 1=Versicolor, 2=Virginica)

# Convert to binary classification: Setosa (1) vs Non-Setosa (0)
y_binary = (y == 0).astype(int)  # 1 if Setosa, 0 otherwise

# Check class balance
unique, counts = np.unique(y_binary, return_counts=True)
print("=" * 70)
print("TASK B1: DATA LOADING AND PREPARATION")
print("=" * 70)
print(f"\nClass Distribution:")
print(f"  Non-Setosa (0): {counts[0]} samples")
print(f"  Setosa (1): {counts[1]} samples")
print(f"  Total: {len(y_binary)} samples")
print(f"  Dataset is {'BALANCED' if abs(counts[0] - counts[1]) < 10 else 'IMBALANCED'}")

# Create a DataFrame for reference
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y_binary
df['class'] = df['target'].map({0: 'Non-Setosa', 1: 'Setosa'})


print("\n" + "=" * 70)
print("TASK B2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Compare feature means between the two classes
print("\nFeature Means Comparison:")
print("-" * 70)
for feature_idx, feature_name in enumerate(iris.feature_names):
    setosa_mean = X[y == 0, feature_idx].mean()
    non_setosa_mean = X[y != 0, feature_idx].mean()
    difference = abs(setosa_mean - non_setosa_mean)
    print(f"{feature_name:25s} | Setosa: {setosa_mean:.2f} | Non-Setosa: {non_setosa_mean:.2f} | Diff: {difference:.2f}")

# Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Bar plot of class distribution
class_counts = pd.Series(y_binary).value_counts().sort_index()
axes[0].bar(['Non-Setosa (0)', 'Setosa (1)'], class_counts.values, color=['#FF6B6B', '#4ECDC4'])
axes[0].set_ylabel('Count')
axes[0].set_title('Class Distribution (Binary)', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# 2. Scatter plot of petal length vs petal width colored by class
colors = ['#FF6B6B' if target == 0 else '#4ECDC4' for target in y_binary]
axes[1].scatter(X[:, 2], X[:, 3], c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].set_title('Petal Length vs Petal Width', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].legend(['Non-Setosa', 'Setosa'], loc='upper left')

plt.tight_layout()
plt.savefig('exploratory_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n[OK] Exploratory analysis plot saved as 'exploratory_analysis.png'")



print("\n" + "=" * 70)
print("TASK B3: MODEL BUILDING AND EVALUATION")
print("=" * 70)

# Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.30, random_state=42, stratify=y_binary
)

print(f"\nData Split:")
print(f"  Training set: {len(X_train)} samples (70%)")
print(f"  Testing set: {len(X_test)} samples (30%)")

# Train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 Non-Setosa  Setosa")
print(f"Actual Non-Setosa     {cm[0, 0]:2d}         {cm[0, 1]:2d}")
print(f"       Setosa         {cm[1, 0]:2d}         {cm[1, 1]:2d}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
