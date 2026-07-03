# Esports Predictive Analytics: Counter-Strike Match Round Outcome Prediction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

## 📌 Project Overview
In competitive esports games like Counter-Strike, being able to predict match outcomes in real time is very useful. Broadcasters can use it to make live coverage more engaging, teams can use it to adjust strategies, and betting platforms can use it to set fair odds. The main challenge is that outcomes depend on many fast-changing factors—such as tactical moves, team finances, and player eliminations—that are all connected in complex ways.

This project builds an end-to-end machine learning classification pipeline using a dataset of **101,832 round snapshots** to predict whether the Counter-Terrorist (CT) or Terrorist (T) faction will secure the round. The pipeline bridges standard statistical approaches with advanced ensemble techniques to optimize overall predictive metrics.

---

## 📊 Key Data Analytics & Strategic Insights

Through data profiling and model feature analysis, several core insights were uncovered regarding match mechanics and competitive balance:

* **The Resource Allocation Paradox:** Economic variables such as `ct_money`, `t_money`, `ct_armor`, and `t_armor` act as primary structural win indicators. Financial superiority maps directly to higher weapon and utility tiers, confirming that early economy management heavily dictates late-round success rates.
* **Spatial Variance:** Map selections (e.g., `de_dust2`, `de_mirage`) provide distinct layouts that fundamentally tilt baseline tactical advantages toward specific factions based on choke points and site entry structures.
* **Linear Discriminant Realities:** Evaluating the feature coefficients using Linear Discriminant Analysis (LDA) proved that active utility counts (like flashbangs and smoke grenades) scale dramatically in significance, holding a comparable weight to a player's raw health pool during live round resolutions.

---

## 🛠️ Repository Architecture

To maintain industry-standard data hygiene and keep the repository clean, the workspace is structured as a modular project:

```text
counterstrike-round-predictor/
│
├── notebooks/
│   └── csgo_round_outcome_predict.ipynb        # CSGO Project
│
├── data/
│   └── csgo_round_snapshots.csv                # Raw Data 
│
├── images/                                     # Asset repository
│   └── target_distribution.png         
│   ├── feature_importance_matrix.png
│   └── model_confusion_matrix.png        
│
├── README.md                                   # Detailed project documentation
└── requirements.txt                            # Project dependencies and environment tracking
```

## ⚙️ Data Pipeline & Technical Setup
### 1. Data Ingestion & Cleaning
- Fixed memory issues when loading mixed data types by adjusting system settings.
- Removed missing values directly in the dataset to keep the data consistent before running calculations.
### 2. Feature Engineering & Preprocessing
- Categorical Conversion: Turned text-based attributes (like map or bomb_planted) into numbers using Label Encoding so models could process them.
- Feature Scaling: Standardized values so large numbers (like money balances) didn’t outweigh small ones (like utility counts).
- Stratified Splitting: Applied stratified splitting (test_size=0.2) to maintain perfectly balanced class distributions across the training and testing sets.
### 3. Model Architecture & Evaluation
- Linear Discriminant Analysis (LDA): Employed as a baseline mathematical reduction model to find the optimal linear combination of features that separates the two winning classes.
- Random Forest Classifier: Deployed as an advanced ensemble technique to capture non-linear relationships, multi-variable interactions, and deep decision boundaries.

## 📈 Performance Matrix & Results
The models were evaluated using overall predictive accuracy alongside detailed precision, recall, and F1-score classification metrics:
```
Predictive Model Architecture          Classification Test Accuracy           Optimization Target Status
Linear Discriminant Analysis (LDA)             ~74.63%                           Baseline Benchmarking
Random Forest Ensemble                         ~87.73%                           Production Candidate
```
## Random Forest Classification Report
                precision    recall  f1-score   support

          CT       0.87      0.89      0.88     12001
           T       0.89      0.87      0.88     12481

    accuracy                           0.88     24482
    macro avg      0.88      0.88      0.88     24482
    weighted avg   0.88      0.88      0.88     24482

    

## 🧠 Key Learnings & Outcomes
- Complete Project Execution: Built the full machine learning process from start to finish, including data cleaning, scaling, reducing dimensions, and evaluating results.

- Model Insights: Learned that simple models like LDA are easy to explain and fast, but more advanced models (like ensembles) capture complex patterns and improved accuracy by about 13% on real game data.

- Clean Coding: Converted messy notebook code into neat, reusable Python functions with clear parameters, making the project easier to maintain and scale.
