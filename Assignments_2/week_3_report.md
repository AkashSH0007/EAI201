Week 3 Report: Model Evaluation and Selection
Project: Predicting FIFA World Cup 2026 Finalists Using Machine Learning

1. Introduction & Evaluation Goals
This report covers the activities for Week 3  corresponding to Task 3: Model Evaluation. The objective of this task is to assess the performance of the two classification models trained.
1.	Logistic Regression
2.	Random Forest Classifier
The evaluation was performed on the unseen test set (20% of the data, 193 matches) to provide a realistic measure of how well each model generalizes. The models were tasked with predicting the match outcome, a multi-class problem with three classes:
•	Class 0: Draw
•	Class 1: Home Team Win
•	Class 2: Away Team Win
2. Model Performance Metrics
The models were evaluated using accuracy, precision, recall, and F1-score. A weighted average is used to fairly account for the class imbalance (i.e., fewer draws than wins).
2.1. Overall Performance Comparison
Model	Accuracy	Weighted Precision	Weighted Recall	Weighted F1-Score	Macro-Avg ROC AUC
Logistic Regression	0.5699	0.4217	0.5699	0.4801	0.6505
Random Forest	0.5078	0.4269	0.5078	0.4479	0.5951
Analysis:
The Logistic Regression model outperformed the Random Forest model on nearly all key metrics, including Accuracy, F1-Score, and ROC AUC. Both models perform significantly better than random guessing (which would be ~33% accuracy) but are still limited in their predictive power.
A key finding from training  was that the Random Forest model achieved 100% training accuracy, which compared to its 50.8% test accuracy, is a definitive sign of severe overfitting. The Logistic Regression model showed less overfitting (58.1% training accuracy vs. 57.0% test accuracy), making it a much more stable and generalizable model.

2.2. Detailed Classification Reports
The full classification reports provide a deeper insight into each model's performance per class.
Logistic Regression
Classification Report (Logistic Regression):
Sl.no	Precision	Recall	F1-score	Support
0	0.00	0.00	0.00	50
1	0.57	0.89	0.70	89
2	0.56	0.57	0.57	54
				
Accuracy			0.57	193
Macro Avg	0.38	0.49	0.42	193
Weighted Avg	0.42	0.57	0.48	193


Random Forest
Classification Report (Random Forest):
             
Sl.No	Precision	Recall	F1-score	support
0	0.13	0.04	0.06	50
1	0.56	0.83	0.67	89
2	0.48	0.41	0.44	54
				
Accuracy			0.51	193
Macro Avg	0.39	0.43	0.39	193
Weighted Avg	0.43	0.51	0.45	193



Key Takeaway (The "Draw" Problem):
Both models are extremely poor at predicting Draws (Class 0). The Logistic Regression model failed to correctly identify a single draw (0.00 precision/recall). The Random Forest model was only slightly better, with 4% recall. This indicates that the features we engineered are not effective at distinguishing the subtle differences that lead to a draw versus a win or loss.



3. Visualization Analysis




3.1. Confusion Matrices
•	Logistic Regression Matrix: The matrix shows that the model heavily favors predicting Class 1 (Home Win), which is the majority class. It misclassifies almost all Draws (Class 0) as home wins (39 out of 50).
•	Random Forest Matrix: This matrix shows a similar, though slightly more scattered, pattern. It also misclassifies the vast majority of draws (39 out of 50) as home wins, confirming its inability to identify this class.

3.2. ROC Curves (One-vs-Rest)
•	The ROC curves visualize the trade-off between the true positive rate and false positive rate for each class.
•	The Macro-Average ROC AUC scores (LR: 0.65, RF: 0.60) confirm the models have some predictive power, as they are both better than the 0.5 "random guess" baseline.
•	The individual curves for Class 0 (Draw) in both plots are very close to the diagonal random-guess line, visually confirming its low AUC score and the models' difficulty in this area.

4. Detailed Model Choice
Based on the evaluation results, the Logistic Regression model is the clear choice to move forward with, despite its limitations.
Justification:
1.	Superior Test Performance: The Logistic Regression model achieved a higher overall Accuracy (57% vs 51%), a better Weighted F1-Score (0.48 vs 0.45), and a significantly better Macro-Average ROC AUC (0.65 vs 0.60) compared to the Random Forest.
2.	Better Generalization (Less Overfitting): The Random Forest model was severely overfit (100% train acc, 51% test acc). The Logistic Regression model was far more stable (58% train acc, 57% test acc), indicating it is more likely to generalize to new, unseen data.
3.	Simplicity & Interpretability: As a linear model, Logistic Regression will be much easier to interpret in Feature Importance by examining its coefficients.
While the chosen model's performance is modest, it is the more robust and reliable of the two. Its primary weakness—the inability to predict draws—is a known challenge.

