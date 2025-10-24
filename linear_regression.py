import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # Convert to pandas DataFrame
df.rename(columns={'MedHouseVal': 'PRICE'}, inplace=True)  # Rename for clarity

# Examine structure
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Price statistics
print("\nPrice Statistics:")
print(df["PRICE"].describe())


# Histogram of house prices
plt.figure(figsize=(7,4))
plt.hist(df["PRICE"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of House Prices (in $100,000s)")
plt.xlabel("Price ($100,000s)")
plt.ylabel("Frequency")
plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# Find most correlated feature with price
price_corr = corr_matrix["PRICE"].drop("PRICE")
most_corr_feature = price_corr.abs().idxmax()
print(f"\nFeature most correlated with price: {most_corr_feature}")
print(f"Correlation value: {price_corr[most_corr_feature]:.3f}")

# Scatter plot of price vs most correlated feature
plt.figure(figsize=(6,4))
sns.scatterplot(x=df[most_corr_feature], y=df["PRICE"], color="blue")
plt.title(f"Price vs {most_corr_feature}")
plt.xlabel(most_corr_feature)
plt.ylabel("Price ($100,000s)")
plt.show()

# Comment on distribution
print("\nSkewness of price distribution:", df["PRICE"].skew())
if df["PRICE"].skew() > 0:
    print("→ The distribution is right-skewed (many houses with lower prices).")
else:
    print("→ The distribution is approximately symmetric or left-skewed.")



# Split data into train and test sets
X = df.drop("PRICE", axis=1)
y = df["PRICE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Scatter plot of actual vs predicted prices
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, color="green")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([0, 5], [0, 5], color="red", linestyle="--")  # reference line
plt.show()

# Interpretation
if r2 > 0.7:
    print(" The R² score indicates good performance.")
elif r2 > 0.5:
    print("The model explains a moderate amount of variance — could be improved.")
else:
    print(" The model has poor performance; try feature scaling or polynomial regression.")

print("\nObservation:")
print("Check if the model underperforms on extreme price ranges (very high or very low prices).")
