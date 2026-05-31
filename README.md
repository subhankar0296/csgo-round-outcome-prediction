# Esports Predictive Analytics: Counter-Strike Match Round Outcome Prediction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

## 📌 Project Overview
In fast-paced, resource-driven competitive esports environments like Counter-Strike, real-time match outcome forecasting is a highly valuable asset for live broadcasting analytics, team strategy optimization, and odds-estimation engines. The core challenge lies in modeling the complex, dynamic interdependencies between tactical milestones, team economic states, and immediate player attrition metrics.

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
├── data/
│   └── csgo_round_snapshots.csv        # Raw Data 
│
├── notebooks/
│   └── csgo_round_predict.ipynb        # Interactive data profiling and charting
│
├── images/                             # Asset repository
│   └── target_distribution.png         
│   ├── feature_importance_matrix.png
│   └── model_confusion_matrix.png        
│
├── src/
│   └── main.py                         # Production-ready, modular execution script
│
├── README.md                           # Detailed project documentation
└── requirements.txt                    # Project dependencies and environment tracking
```

## ⚙️ Data Pipeline & Technical Architecture
### 1. Data Ingestion & Sanitization
Addressed low-memory mixed-datatype warnings during data ingestion directly via optimized interpreter parameters.
Handled missing data entries programmatically by enforcing a clean-in-place sequence (dropna) to guarantee matrix integrity before downstream mathematical operations.
### 2. Feature Engineering & Preprocessing
- Categorical Conversion: Transformed domain-specific nominal attributes (map, bomb_planted) into model-ready numerical values using robust Label Encoding.
- Feature Scaling: Normalized wide feature variances (e.g., thousands of dollars in economy balances vs. single-digit utility counts) via a StandardScaler to eliminate magnitude and scale bias.
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

    

## 🧠 Key Learnings & Engineering Outcomes
- End-to-End Execution: Developed a complete machine learning lifecycle including pipeline ingestion, structured data imputation, feature scaling, mathematical dimensionality reduction, and evaluation matrix generations.

- Scaling Realities: Confirmed that while simpler models like LDA offer excellent mathematical explainability and fast inference speeds, complex ensemble methods provide the non-linear boundaries required to secure a significant ~13% accuracy boost on live game telemetry.

- Clean Code Standards: Refactored exploratory notebook blocks into modular, reusable Python functions paired with explicit parameter tracking to ensure script maintainability and scalability.
