# Comprehensive Esports Analytics: Counter-Strike Match Round Outcome Prediction

📌 Project Overview
In fast-paced, resource-driven competitive esports environments like Counter-Strike, real-time match outcome forecasting is a highly valuable asset for broadcasting analytics, team strategy optimization, and live-odds estimation. The core challenge lies in the complex, dynamic interdependencies between tactical choices, financial states, and real-time attrition metrics.

This project builds an end-to-end machine learning classification pipeline using a dataset of 101,832 round snapshots to predict whether the Counter-Terrorist (CT) or Terrorist (T) faction will secure the round. The pipeline evaluates multiple models, bridging a baseline statistical reduction approach with an advanced ensemble method to optimize prediction metrics.

📊 Key Data Analytics & Business Insights
Through rigorous exploratory data analysis and modeling feature evaluation, several core insights were uncovered regarding competitive balance:

The Resource Allocation Paradox: Economic features such as ct_money, t_money, ct_armor, and t_armor act as the primary structural win indicators. Financial superiority maps directly to higher weapon and utility tiers, confirming that early economy management dictates late-round success rates.

Spatial Variance: Map selections (e.g., de_dust2, de_mirage) provide distinct layouts that fundamentally tilt tactical advantages toward specific factions based on choke points and site entry structures.

Linear Discriminant Realities: Flattening the feature coefficients using Linear Discriminant Analysis (LDA) proved that active utility counts (like flashbangs and smoke grenades) scale dramatically in significance, often holding a comparable weight to a player's raw health pool in live round resolutions.

⚙️ Data Pipeline & Technical Architecture
1. Data Ingestion & Sanitization
Addressed low-memory mixed-datatype warnings during ingestion directly via optimized parameters.

Handled missing records programmatically by enforcing an in-place cleanup sequence to guarantee array integrity before downstream math operations.

2. Feature Engineering & Preprocessing
Categorical Conversion: Transformed domain-specific nominal attributes (map, bomb_planted) into model-ready numerical values using robust Label Encoding.

Feature Scaling: Normalized wide feature variances (e.g., thousands of dollars in economy vs. single-digit utility counts) via StandardScaler to eliminate magnitude bias.

Stratified Splitting: Applied stratified splitting (test_size=0.2) to maintain perfectly balanced class distributions across training and validation testing sets.

3. Model Architecture & Dimensionality Reduction
Linear Discriminant Analysis (LDA): Employed as a baseline mathematical reduction model to find the optimal linear combination of features that separates the two winning classes.

Random Forest Classifier: Deployed as an advanced ensemble technique to capture non-linear relationships and intricate feature interactions.

📈 Performance Matrix & Results
The models were evaluated using overall predictive accuracy alongside detailed precision, recall, and F1-score classification metrics:
Predictive Model ArchitectureClassification Test AccuracyOptimization Target StatusLinear Discriminant Analysis (LDA)~75.5%Baseline BenchmarkingRandom Forest Ensemble~84.3%Production Candidate

# Random Forest Classification Report
precision    recall  f1-score   support

      CT Win       0.84      0.84      0.84     10174
       T Win       0.84      0.84      0.84     10193

    accuracy                           0.84     20367
   macro avg       0.84      0.84      0.84     20367
weighted avg       0.84      0.84      0.84     20367

The ensemble model achieves excellent predictive balance with uniform precision and recall across both factions (~84%), minimizing false-positive errors for both defensive and offensive match states.

🧠 Key Learnings & Engineering Outcomes
End-to-End Execution: Developed a complete machine learning lifecycle including ingestion pipelines, structured imputation, scaling, mathematical dimensionality reduction, and evaluation matrix generations.

Scaling Realities: Confirmed that while simpler models like LDA offer excellent mathematical explainability and fast inference, complex ensemble methods provide the non-linear boundaries required to squeeze out an additional ~9% accuracy boost on live game telemetry.

Clean Code Standards: Refactored exploratory code blocks into modular, reusable Python functions paired with explicit parameter tracking to ensure script maintainability and scalability.
