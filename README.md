----Student Performance Prediction----

Overview:
Predict student pass/fail outcomes in the Mathematics course using demographic, academic, and behavioral features. The model uses a Multi-Layer Perceptron (MLP) implemented with TensorFlow/Keras.

Dataset:
Source: UCI ML Repository – Student Performance
Records: 395 students
Features: 33 (numeric & categorical)
Target: pass_fail (binary, derived from final grade G3 ≥10 → Pass)

Workflow:
EDA & Visualization: Histograms, boxplots, scatterplots, correlation heatmaps.
Preprocessing:
-Numeric: Median imputation & standardization
-Categorical: Mode imputation & one-hot encoding
Model:
-MLP with 64 → 32 hidden neurons, ReLU + Dropout
-Sigmoid output for binary classification
Evaluation: Accuracy, Confusion Matrix, ROC-AUC, Classification Report

Key Insights:
Top predictors: G1, G2, study time, absences
Gender: Minor differences in grades
MLP shows high accuracy and effective pass/fail prediction

Future Work:
Include behavioral/psychological features
Compare with ensemble models (Random Forest, XGBoost)
Build interactive dashboards
Use Explainable AI (SHAP/LIME)

Requirements:
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras

Usage
--python student_mat_passfail_mlp.py--
Loads and preprocesses data
Trains MLP model
Generates evaluation metrics and plots
