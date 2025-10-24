import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df.head()

print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

df.describe()

sns.countplot(data=df, x="Survived", palette="Set2")
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.show()

# Sex vs Surviva
sns.countplot(data=df, x="Sex", hue="Survived", palette="husl")
plt.title("Survival by Sex")
plt.show()

# Pclass vs Survival
sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set1")
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution
sns.histplot(data=df, x="Age", bins=30, kde=True, color="teal")
plt.title("Age Distribution")
plt.show()

# Fare distribution
sns.boxplot(data=df, x="Survived", y="Fare", palette="cool")
plt.title("Fare vs Survival")
plt.show()

df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill missing 'Fare' with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop irrelevant columns
df.drop(columns=['PassengerId', 'Ticket', 'Cabin'], inplace=True)

df.isnull().sum()  # Check again


# Create FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Extract Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplify rare titles
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# Drop the original Name column
df.drop(columns=['Name'], inplace=True)

df[['Sex', 'Embarked', 'Title']].head()

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

df.head()


# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("Data ready for modeling!")
X_train.info()
