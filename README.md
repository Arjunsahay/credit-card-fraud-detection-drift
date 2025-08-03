# 🛡️ Real-Time Credit Card Fraud Detection with Concept Drift Handling

This project simulates a real-time fraud detection system that processes streaming-style batches of credit card transactions, detects concept drift in the data, and retrains the model automatically when needed.

---

## 📌 Overview

- **Dataset**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Skills Used**: `pandas`, `scikit-learn`, `xgboost`, `statsmodels`
- **Goal**: Detect fraud in real-time batches while adapting to evolving patterns (concept drift)
- **Challenge**: Handle severe class imbalance and maintain high recall over time
- **Edge**: Integrates a statistical drift detector (KS test) with a live retraining loop

---

## 📂 Project Structure

REAL TIME CREDIT CARD FRAUD DETECTION/
├── data/
│ └── creditcard.csv # Original dataset (download manually from Kaggle)
├── src/
│ ├── generate_batches.py # Loads and chunks data into streaming-style batches
│ ├── preprocessing.py # StandardScaler logic
│ ├── drift_detector.py # KS-test drift detection
│ ├── model_trainer.py # XGBoost training and evaluation
│ ├── real_time_simulator.py # Main streaming + drift detection + retraining loop
│ └── config.py # Constants like batch size, thresholds, etc.
├── main.py # Project entrypoint
└── requirements.txt # Python dependencies


---

## 📊 Dataset Description

- **Total rows**: 284,807 transactions
- **Frauds**: Only 492 → **severe class imbalance**
- **Features**:  
  - `V1` to `V28`: PCA-transformed features (anonymized)  
  - `Amount`, `Time`: raw features (we log-transform `Amount`)  
  - `Class`: target (1 = fraud, 0 = not fraud)

---

## 🧠 Core Concepts

### 1. Real-Time Batching
Simulates incoming transactions using fixed-size batches (e.g., 5,000 rows). Each batch is treated as a new time window of data.

### 2. Model Training (XGBoost)
- Handles imbalance using `scale_pos_weight = N_neg / N_pos`
- Evaluates on precision, recall, and F1 after each batch

### 3. Concept Drift Detection (KS Test)
- Uses **Kolmogorov-Smirnov test** to compare feature distributions in:
  - `reference_data` (initial batch)
  - `current_data` (new batch)
- If p-value < 0.05 for any feature → **drift is flagged**
- Drifted model is **retrained** on the new batch

---

## 🏁 How It Works

### 🔁 Workflow:
1. Load full dataset
2. Shuffle and split into batches
3. For each batch:
   - Preprocess using `StandardScaler`
   - If first batch: train model and save as reference
   - Else:
     - Check for drift (KS test)
     - Retrain if drift detected
   - Predict and evaluate on current batch

### 🖥️ Console Output Example:

📦 Processing Batch 55
⚠️ Drift detected in feature: V3 (p=0.0443)
🔄 Drift detected — retraining model
C:\Users\Krish\AppData\Local\Programs\Python\Python312\Lib\site-packages\xgboost\training.py:183: UserWarning: [09:01:43] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:738:
Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
📊 Evaluation:
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000      4991
           1     1.0000    1.0000    1.0000         9

    accuracy                         1.0000      5000
   macro avg     1.0000    1.0000    1.0000      5000
weighted avg     1.0000    1.0000    1.0000      5000


📦 Processing Batch 56
⚠️ Drift detected in feature: V4 (p=0.0469)
🔄 Drift detected — retraining model
C:\Users\Krish\AppData\Local\Programs\Python\Python312\Lib\site-packages\xgboost\training.py:183: UserWarning: [09:01:43] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:738:
Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
📊 Evaluation:
              precision    recall  f1-score   support

           0     1.0000    0.9998    0.9999      4992
           1     0.8889    1.0000    0.9412         8

    accuracy                         0.9998      5000
   macro avg     0.9444    0.9999    0.9705      5000
weighted avg     0.9998    0.9998    0.9998      5000


