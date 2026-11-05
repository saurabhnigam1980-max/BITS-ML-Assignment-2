
# BITS-ML-Assignment-2

# Machine Learning Model Classification & Interactive Deployment

# Chronic Disease & Diabetes Risk Evaluation Dashboard

## a. Problem Statement
The primary objective of this project centers on optimizing a predictive classification engine to evaluate clinical early-stage Diabetes Risk Susceptibility based on multi-axial patient health records. Utilizing an evaluation matrix containing 1,000 unique data records across 12 distinct numeric lifestyle and clinical features, the workflow accurately models binary classification mappings (0: Healthy / Stable, 1: High Diabetes Risk Profile). The production architecture introduces a standalone web interface built on Streamlit that dynamically loads pre-trained serialized model artifacts, enabling immediate inference on uploaded evaluation batches without triggering a model retraining cycle.

## b. Dataset Description
* **Total Instances:** 1,000 samples.
* **Feature Dimensions:** 12 independent continuous numeric features mapped to explicit clinical indicators.
* **Informative Features:** 9 features contain highly discriminative signals directly correlated with the target label.
* **Redundant Features:** 3 features are linear combinations of informative features, simulating realistic multicollinearity challenges.
* **Target Variable:** Binary column (target) tracking patient metabolic category (0 = Stable, 1 = Positive Diabetes Risk).

## c. GitHub Repository Link
* **Repository URL:** https://github.com
https://github.com/saurabhnigam1980-max/BITS-ML-Assignment-2

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
* **Logistic Regression:** Establishes a rigid linear baseline architecture mapping (85.50% accuracy). It displays minor underfitting when mapping complex, non-linear dependencies formed by overlapping risk combinations like HighBP and HighChol.
* **Decision Tree:** Captures explicit conditional variable routing boundaries (86.00% accuracy) with a strong recall capability of 86.17%. However, its constrained AUC profile (0.8601) points to hard probabilistic step limitations typical of unpruned leaf splits.
* **K-Nearest Neighbors (KNN):** Adapts tightly to localized density clusters within the multi-dimensional feature space, generating excellent validation results (93.00% accuracy, 0.9468 Recall). Its distance metrics operate highly effectively across the scaled continuous vectors of BMI and Age.
* **Naive Bayes (Gaussian):** Displays the weakest overall evaluation footprint (79.00% accuracy, 0.5789 MCC score). This notable performance drop is directly driven by its strict underlying assumption of conditional feature independence, which is systematically violated by structural lifestyle associations engineered between Fruits, Veggies, and PhysActivity.
* **Support Vector Machine (SVM):** Equalizes with KNN on overall classification accuracy (93.00%) but sets the highest standalone framework benchmark for confidence and reliability. It yields an outstanding Precision score of 95.45% alongside a domain-leading AUC metric of 0.9847.

### Overall Dataset Winner
The overall selection for this dataset is the **Support Vector Machine (SVM)**. Configured with a Radial Basis Function (RBF) kernel, it is selected as the primary clinical deployment model. In medical informatics pipelines, minimizing false positives is critical to avoid misallocating clinical resources. By providing the highest validation Precision (95.45%) and near-perfect class separation capability (AUC: 0.9847), the SVM boundary guarantees the most reliable generalizability for clinical deployment.