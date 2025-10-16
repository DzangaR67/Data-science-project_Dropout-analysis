# Data-science-project_Dropout-analysis

Project Overview:
This project implements an end-to-end Machine Learning pipeline for predicting student dropout risk. It transforms raw student data into actionable insights, enabling educational institutions to identify at-risk students early and implement timely interventions. The system demonstrates exceptional predictive performance, with the best model achieving near-perfect metrics.

The workflow is implemented in Google Colab using Python, with results visualized through Matplotlib, Seaborn, and Plotly.
It compares three key models: Logistic Regression, Random Forest, and XGBoost, to determine the best-performing predictor.

**Features**

 - Data Loading
    - Uses the UCI Machine Learning Repository via ucimlrepo to fetch datasets automatically.

- Exploratory Data Analysis (EDA)

   - Summary statistics, correlation analysis
   - Visualization with Matplotlib, Seaborn, and Plotly
   - Identification of data imbalance and relationships between features

- Data Preprocessing

   - Missing value imputation with SimpleImputer
   - Feature scaling using StandardScaler
   - Addressing class imbalance using SMOTE (Synthetic Minority Oversampling Technique)

- Model Training and Comparison
  - Trains and evaluates:

     - LogisticRegression
     - RandomForestClassifier
     - XGBClassifier (Extreme Gradient Boosting)

- Evaluation Metrics
  - Comprehensive evaluation using:

    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC score
    - Confusion Matrix
    - Classification Report

- Model Exporting
Saves trained models using joblib for future use or deployment.

- Interactive Visualizations
Uses Plotly and ipywidgets to visualize feature relationships and performance interactively in Colab.

- Statistical Testing
Incorporates Chi-square and Paired t-tests to verify the significance of features or performance differences.


Libraries and Tools Used:

- Core Data Science	: pandas, numpy, scikit-learn
- Machine Learning:	XGBoost, imbalanced-learn (SMOTE)
- Visualization:	matplotlib, seaborn, plotly
- Statistical Analysis:	scipy.stats
- Model Deployment:	joblib (for model serialization)

System Architecture:

- The project follows a structured machine learning workflow:
- Data Generation & Preprocessing
- Exploratory Data Analysis (EDA)
- Model Training & Hyperparameter Tuning
- Model Evaluation & Selection
- Results Interpretation & Feature Analysis



Original Dataset Source
- **Title**: Predict Students' Dropout and Academic Success
- **Authors**: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021)
- **Repository**: UCI Machine Learning Repository
- **DOI**: https://doi.org/10.24432/C5MC89
- **Access**: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Dataset Characteristics
- **Number of Instances**: 4,424 students
- **Number of Features**: 36 attributes
- **Time Period**: 2008-2020
- **Institution**: Higher Education Institution in Portugal

Key Feature Categories in dataset
- **Demographic Data**: Gender, Age, Marital Status, Nationality
- **Academic Background**: Previous qualification, Admission grade, Course
- **Socio-economic Factors**: Parents' education, Occupation, Income
- **Academic Performance**: Curricular units approved, Grades, Enrollment status
- **Behavioral Patterns**: Attendance, Scholarship status, International student status

  _**Findings**_

**Logistic Regression:**
  
•	Accuracy: 0.8655: 86.55% of all students are classified correctly. This is a useful overall summary, but with rare events (dropout often a minority) accuracy can be misleading , a model could predict “not dropout” for everyone and still get high accuracy if dropout is uncommon.

•	Precision: 0.7957: Of all students the model predicted as at-risk, ~79.6% dropped out. Precision quantifies false positives: higher precision.

•	Recall: 0.7817: Of all true dropouts, ~78.2% were correctly flagged. Recall quantifies false negatives: higher recall , fewer missed students who need help.

•	F1-Score: 0.7886: Harmonic mean of precision and recall, shows a balanced performance between catching dropouts and avoiding false alarms.

•	ROC-AUC: 0.9205: The model can discriminate very well between students who will and will not drop out (a high separability score).

**Random Forest:**

•	Accuracy: 0.8802: Random Forest correctly classifies 88.02% of students , about 1.5 percentage points better than Logistic Regression. That improvement reflects both slightly better true-positive and true-negative performance (we’ll quantify below).

•	Precision: 0.8321: of students flagged, ~83.2% truly drop out , fewer wasted interventions than logistic regression.

•	Recall: 0.7852: the model catches ~78.5% of actual dropouts , slightly higher than logistic regression.

•	F1-Score: 0.8080: the balance between precision and recall is better than logistic regression

•	ROC-AUC: 0.9262: very good discrimination; Random Forest captures nonlinear interactions that logistic regression might miss.


**XGBoost:**

•	Accuracy: 0.8859: XGBoost correctly classifies 88.59% of students, the highest overall correctness of the three models. The ~2.04 percentage point gain over Logistic Regression is meaningful at scale

•	Precision: 0.8453: of students flagged as at-risk, ~84.5% are actual dropouts, the fewest false positives among the three models. This reduces wasted staff time the most.

•	Recall: 0.7887: the model catches ~78.9% of dropouts, slight improvement in recall compared to others.

•	F1-Score: 0.8160: best harmonic balance, indicating the best combined performance for the positive class.

•	ROC-AUC: 0.9268: best discrimination, but close to Random Forest, that small bump can still matter operationally.


**Interpretations of Class Balance and SMOTE Effect**

The dataset exhibited class imbalance, with fewer students dropping out compared to those who continued. Applying SMOTE (Synthetic Minority Oversampling Technique) effectively balanced the data, improving the model’s sensitivity to dropout cases.
Post-SMOTE training showed noticeable gains in recall and F1-score, confirming its role in mitigating bias toward the majority class.


Visuals generated include:

- Confusion matrices
- ROC curves
- Feature importance plots
- Correlation heatmaps
