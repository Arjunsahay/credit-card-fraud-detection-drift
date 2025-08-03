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
