# BITS-ML-Assignment-2
# Machine Learning Model Classification & Interactive Deployment

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare multiple standard machine learning classification models on a high-dimensional dataset containing complex multi-feature interactions. Using a structured dataset with 12 distinct numeric features and 1,000 instances, the goal is to accurately predict a binary target variable (target ∈ {0, 1}). The final phase requires developing and deploying a standalone web interface that can load these saved models dynamically to evaluate unseen test sets without triggering a retraining phase.

## b. Dataset Description
* **Total Instances:** 1,000 samples.
* **Feature Dimensions:** 12 independent continuous numeric features (feature_1 through feature_12).
* **Informative Features:** 9 features contain highly discriminative signals directly correlated with the target label.
* **Redundant Features:** 3 features are linear combinations of informative features, simulating realistic multicollinearity challenges.
* **Target Variable:** Binary column (target) with class values balanced between 0 and 1.

## c. GitHub Repository Link
* **Repository URL:** https://github.com](https://github.com/saurabhnigam1980-max/BITS-ML-Assignment-2

## d. Models Used & Evaluation Metrics Comparison Table
All five classification models were trained on the identical training split (800 rows) and evaluated on the identical unseen validation split (test_data.csv, 200 rows).

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|:-----------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression    | 0.8550   | 0.8927 | 0.8652    | 0.8191 | 0.8415 | 0.7090 |
| Decision Tree          | 0.8600   | 0.8601 | 0.8438    | 0.8617 | 0.8526 | 0.7195 |
| KNN                    | 0.9300   | 0.9786 | 0.9082    | 0.9468 | 0.9271 | 0.8605 |
| Naive Bayes            | 0.7900   | 0.8778 | 0.8095    | 0.7234 | 0.7640 | 0.5789 |
| Support Vector Machine | 0.9300   | 0.9847 | 0.9545    | 0.8936 | 0.9231 | 0.8606 |

## e. Model Performance Observations & Analysis
### Detailed Model Analysis
* **Logistic Regression:** Demonstrates dependable performance with an accuracy of 85.50% and an MCC of 0.7090. As a linear baseline algorithm, it effectively maps major trends but slightly struggles with highly non-linear feature interactions hidden within the informative columns.
* **Decision Tree:** Achieves an accuracy of 86.00% along with a strong Recall value of 86.17%. While it captures feature splits cleanly, its lower AUC score (0.8601) relative to its accuracy points to hard probabilistic boundary limitations common in unpruned structures.
* **K-Nearest Neighbors (KNN):** Yields strong predictive output across all metrics (93.00% accuracy, 0.9786 AUC, and 0.9468 Recall). By focusing on localized spatial clusters, it successfully adapts to the density distribution of this specific continuous feature space.
* **Naive Bayes (Gaussian):** Exhibits the lowest overall performance profile (79.00% accuracy, 0.5789 MCC). This drop is primarily driven by its foundational assumption of feature independence, which is systematically violated by the 3 redundant, highly correlated features introduced in our dataset.
* **Support Vector Machine (SVM):** Matches KNN at 93.00% accuracy but sets the highest benchmark for classification confidence with an outstanding Precision of 95.45% and a top AUC score of 0.9847.

### Overall Dataset Winner
The overall selection for this dataset is the **Support Vector Machine (SVM)**. While both KNN and SVM tied for the highest structural accuracy (93.00%) and near-identical MCC scores (~0.860), SVM provides superior robustness. Its dominant AUC score of **0.9847** highlights near-perfect class separation capability, and its exceptionally high Precision (95.45%) minimizes costly false-positive misclassifications far better than any other model.
