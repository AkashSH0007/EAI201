Week 2 Report: Model Building and Training

1. Introduction
This report summarizes the activities undertaken during Week 2. Focusing on Task 2: Model Building and Training. Following the successful data preparation. Next step involved selecting appropriate features, preprocessing the data, implementing the required classification models (Logistic Regression and Random Forest), and performing initial training.
2. Code Overview
All model building and training steps were implemented in Python using the scikit-learn library. The primary script used for this phase is model_training.py. This script loads the final dataset created in Week 1 (world_cup_final_dataset.csv), performs preprocessing, splits the data, trains the models, and includes initial evaluation steps.
3. Feature Selection
The following 13 features were selected as inputs (X) for the models:
1.	neutral: Whether the match was played at a neutral venue (Boolean converted to 0/1).
2.	home_team_avg_age: Average age of the home team.
3.	home_team_total_market_value: Total market value of the home team.
4.	home_team_avg_experience: Average total minutes played for the home team.
5.	away_team_avg_age: Average age of the away team.
6.	away_team_total_market_value: Total market value of the away team.
7.	away_team_avg_experience: Average total minutes played for the away team.
8.	home_team_rank: FIFA Rank of the home team.
9.	home_team_rank_points: FIFA Points of the home team.
10.	away_team_rank: FIFA Rank of the away team.
11.	away_team_rank_points: FIFA Points of the away team.
12.	home_team_win_rate: Historical World Cup win rate for the home team.
13.	away_team_win_rate: Historical World Cup win rate for the away team.
Excluded Feature:
•	goal_difference: This feature was initially included but subsequently removed. It is calculated from the match scores (home_score - away_score), which directly determine the match_outcome target variable. Including it led to perfect but unrealistic model performance (data leakage). Excluding it provides a more realistic modeling challenge.


Target Variable:
•	match_outcome: The target variable (y) representing the result of the match (0=Draw, 1=HomeWin, 2=AwayWin).
4. Preprocessing
Before training the models, preprocessing was applied to the selected features:
•	Scaling: Numerical features exhibit a wide range of values (e.g., market value vs. average age). To ensure that features with larger values do not disproportionately influence models sensitive to feature scale (like Logistic Regression), StandardScaler from scikit-learn was used.
o	The scaler was fit only on the training data (X_train) to prevent data leakage from the test set.
o	Both the training data (X_train) and the test data (X_test) were then transformed using the fitted scaler, resulting in X_train_scaled and X_test_scaled.
o	The fitted scaler object was saved (scaler.joblib) for potential use later when predicting on new, unseen data (e.g., for the 2026 predictions).
•	Encoding: The neutral feature (Boolean) was converted to an integer (0/1). No other categorical features were selected for this initial modeling phase, avoiding the need for further encoding like One-Hot Encoding at this stage.
5. Model Implementation
Two classification models were implemented.
1.	Logistic Regression:
o	Chosen as a standard baseline model for classification. It's relatively simple and provides interpretable coefficients.
o	Implemented using sklearn.linear_model.LogisticRegression.
o	Configured for multi-class classification using the  (One-vs-Rest) strategy.
o	max_iter was increased to 1000 to ensure convergence.
o	Trained on the scaled training data (X_train_scaled, y_train).
2.	Random Forest Classifier:
o	Chosen as a more complex, tree-based ensemble model capable of capturing non-linear relationships.
o	Implemented using sklearn.ensemble.RandomForestClassifier.
o	Configured with default parameters (e.g., n_estimators=100).
o	Trained on the scaled training data (X_train_scaled, y_train).


6. Validation Approach
To ensure a fair evaluation of how the models generalize to unseen data, a standard train-test split strategy was employed:
•	The dataset (X, y) was split using sklearn.model_selection.train_test_split.
•	80% of the data was allocated for training (X_train, y_train).
•	20% of the data was held out for testing (X_test, y_test).
•	random_state=42 was used to ensure the split is reproducible.
•	stratify=y was used to maintain the same proportion of outcome classes (Draw/HomeWin/AwayWin) in both the training and testing sets, which is important for potentially imbalanced classes.

7. Initial Training Summary & Hyperparameter Tuning
•	Both models were successfully trained on the prepared data.
•	Logistic Regression Training Accuracy: ~0.5811
•	Random Forest Training Accuracy: 1.0000
The perfect training accuracy for the Random Forest strongly suggests overfitting. The model likely memorized the training data. The Logistic Regression's training accuracy is much lower, indicating less overfitting but potentially lower predictive power overall.
Hyperparameter Tuning: No explicit hyperparameter tuning (e.g., using GridSearchCV or RandomizedSearchCV) was performed in this initial training phase. Default parameters were used for both models. Tuning could be a potential next step to optimize performance, especially for the Random Forest to reduce overfitting (e.g., by adjusting max_depth, min_samples_split, n_estimators).
8. Conclusion
Week 2 successfully covered the requirements. Final dataset was prepared for modeling, including feature selection and scaling. Two classification models, Logistic Regression and Random Forest, were implemented and trained using a train-test split validation strategy. Initial training accuracy suggests potential overfitting, particularly for the Random Forest model. The trained models (logistic_regression_model.joblib, random_forest_model.joblib) and the scaler (scaler.joblib) have been saved. 

