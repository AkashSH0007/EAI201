import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file name from Task 1
TASK_1_OUTPUT_FILE = 'task_1_integrated_data.csv'

print(f"--- Task 2: Delta_eda_and_cleaning ---")

# Load the dataset
try:
    df = pd.read_csv(TASK_1_OUTPUT_FILE)
    print(f"Successfully loaded '{TASK_1_OUTPUT_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{TASK_1_OUTPUT_FILE}' not found. Cannot proceed with Task 2.")
    # Stop execution if the file isn't found
    raise

# Define original numeric features + engineered features
# These are the features we'll use for variance and correlation
numeric_features = [
    'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator',
    'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail',
    'domestic', 'catsize', 'ecosystem_type', 'predator_score'
]
# Ensure all features are in the dataframe
features_for_analysis = [f for f in numeric_features if f in df.columns]
if len(features_for_analysis) != 18:
    print(f"Warning: Expected 18 numeric/engineered features, but found {len(features_for_analysis)}.")

# --- A) Visualizations ---
print("\n--- A) Generating Visualizations ---")

# 1. Horizontal bar chart (Class distribution)
print("1. Generating Class Distribution bar chart...")
plt.figure(figsize=(12, 8))
# Get value counts, sorted
class_counts = df['Class_Type'].value_counts(ascending=True)
total_animals = class_counts.sum()
# Create horizontal bar plot
ax = class_counts.plot(kind='barh', color='steelblue')
ax.set_title('Animal Class Distribution', fontsize=16)
ax.set_xlabel('Number of Animals', fontsize=12)
ax.set_ylabel('Class Type', fontsize=12)
# Add percentage labels
for i, count in enumerate(class_counts):
    percentage = (count / total_animals) * 100
    ax.text(count + 0.1, i, f'{percentage:.1f}% ({count})', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('class_distribution_bar_chart.png')
print("Saved 'class_distribution_bar_chart.png'")
plt.close()


# 2. Box plot (Engineered features vs classes)
print("2. Generating Engineered Features vs. Class box plots...")
# Plot 2a: ecosystem_type
plt.figure(figsize=(12, 7))
sns.boxplot(x='Class_Type', y='ecosystem_type', data=df)
sns.stripplot(x='Class_Type', y='ecosystem_type', data=df, color='black', alpha=0.5, jitter=0.2)
plt.title('Ecosystem Type vs. Class Type', fontsize=16)
plt.xlabel('Class Type', fontsize=12)
plt.ylabel('Ecosystem Type', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ecosystem_type_vs_class_boxplot.png')
print("Saved 'ecosystem_type_vs_class_boxplot.png'")
plt.close()

# Plot 2b: predator_score
plt.figure(figsize=(12, 7))
sns.boxplot(x='Class_Type', y='predator_score', data=df)
sns.stripplot(x='Class_Type', y='predator_score', data=df, color='black', alpha=0.5, jitter=0.2)
plt.title('Predator Score vs. Class Type', fontsize=16)
plt.xlabel('Class Type', fontsize=12)
plt.ylabel('Predator Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('predator_score_vs_class_boxplot.png')
print("Saved 'predator_score_vs_class_boxplot.png'")
plt.close()


# 3. Scatter plot matrix (Diet vs features)
print("3. Generating Scatter Plot Matrix (Pairplot)...")
# We'll use a subset of features for readability, with 'diet' as the hue
pairplot_features = ['ecosystem_type', 'predator_score', 'legs', 'aquatic', 'milk', 'diet']
# Handle missing values for the plot (if any, though 'diet' should be filled)
pairplot_df = df[pairplot_features].fillna('unknown')
sns.pairplot(pairplot_df, hue='diet', palette='tab10')
plt.suptitle('Pairplot of Features, Colored by Diet', y=1.02, fontsize=16)
plt.savefig('diet_vs_features_pairplot.png')
print("Saved 'diet_vs_features_pairplot.png'")
plt.close()


# 4. Correlation heatmap with clustering
print("4. Generating Correlation Clustermap...")
corr_matrix = df[features_for_analysis].corr()
plt.figure(figsize=(14, 12))
# Use clustermap
cluster_map = sns.clustermap(
    corr_matrix, 
    annot=True, 
    fmt='.2f', 
    cmap='vlag', 
    linewidths=.5,
    figsize=(18, 18) # Size for the clustermap
)
# Adjust the main title
cluster_map.fig.suptitle('Clustered Correlation Heatmap of Features', fontsize=20, y=1.03)
plt.savefig('correlation_clustermap.png', bbox_inches='tight')
print("Saved 'correlation_clustermap.png'")
plt.close()


# --- B) Metrics ---
print("\n--- B) Calculating Metrics ---")

# 1. Class imbalance ratio
print("\n1. Class Imbalance Ratio:")
class_counts = df['Class_Type'].value_counts()
largest_class_size = class_counts.max()
smallest_class_size = class_counts.min()
imbalance_ratio = largest_class_size / smallest_class_size
print(f"  Largest Class (Mammal): {largest_class_size}")
print(f"  Smallest Class (Amphibian): {smallest_class_size}")
print(f"  Class Imbalance Ratio = {largest_class_size} / {smallest_class_size} = {imbalance_ratio:.2f}")


# 2. Low variance features
print("\n2. Low Variance Features (variance < 0.01):")
variances = df[features_for_analysis].var()
low_variance_features = variances[variances < 0.01].index.tolist()
if low_variance_features:
    for feature in low_variance_features:
        print(f"  - {feature}: {variances[feature]:.4f}")
else:
    print("  No features found with variance < 0.01.")


# 3. Highly correlated pairs
print("\n3. Highly Correlated Pairs (|corr| > 0.8):")
corr_matrix = df[features_for_analysis].corr().abs()
# Stack the matrix and reset index to turn it into a list of pairs
corr_pairs = corr_matrix.stack().reset_index()
corr_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']
# Filter out self-correlation (Feature_1 == Feature_2)
corr_pairs = corr_pairs[corr_pairs['Feature_1'] != corr_pairs['Feature_2']]
# Filter for high correlation
highly_correlated = corr_pairs[corr_pairs['Correlation'] > 0.8]
# To avoid duplicates (e.g., (A,B) and (B,A)), we sort features
highly_correlated['sorted_pair'] = highly_correlated.apply(
    lambda row: tuple(sorted((row['Feature_1'], row['Feature_2']))), 
    axis=1
)
highly_correlated = highly_correlated.drop_duplicates('sorted_pair').drop(columns='sorted_pair')

if not highly_correlated.empty:
    for index, row in highly_correlated.iterrows():
        print(f"  - ({row['Feature_1']}, {row['Feature_2']}): {row['Correlation']:.4f}")
else:
    print("  No highly correlated pairs found with |corr| > 0.8.")

print("\n--- Task 2 Complete ---")