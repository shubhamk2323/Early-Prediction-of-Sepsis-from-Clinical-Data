# Sepsyd Replication: Sepsis Prediction Pipeline

This repository contains the replication of the **Sepsyd Model** (Rank 3 in the PhysioNet 2019 Early Prediction of Sepsis Challenge). The pipeline trains an XGBoost classifier on massive, real-world ICU patient records to predict Sepsis onset hours before it occurs.

## 🚀 Project Overview
Sepsis is a severe clinical condition with high mortality rates if not detected early. This project focuses on analyzing messy, real-world intensive care unit (ICU) data (vital signs, laboratory test results) from over 40,000 patients. 

We downloaded and merged over 40k individual `.psv` files into a single, massive 1.7+ million row dataset (`combined_sepsis_data.csv`). We then implemented the official feature extraction and modeling logic from the Sepsyd team's methodology.

## 🧠 Key Features & Technical Highlights

* **Z-Score Normalization:** Normalized 40 distinct clinical features using the exact population baseline means and standard deviations discovered by the Sepsyd team.
* **Log Scaling:** Applied logarithmic scaling to highly skewed laboratory results (like Bilirubin and Creatinine) to stabilize variance.
* **Delta Calculations:** Calculated the exact hour-to-hour velocity of every vital sign and lab value (`Current Hour - Previous Hour`).
* **Temporal Array Flattening:** Implemented a 6-hour sliding window architecture. Instead of just taking the mean, we shifted the entire clinical array up to `T-5`, flattening 6 hours of patient history into a single 720-dimensional feature vector per hour.
* **XGBoost Classifier:** Trained an XGBoost model using the precise hyperparameters from the Sepsyd paper (e.g., `scale_pos_weight: 40`, `max_depth: 4`, `n_estimators: 30`).

## 📊 Evaluation & Metrics
The pipeline includes an Ultra Comprehensive Sepsis Evaluation Report to evaluate performance under extreme class imbalance. Typical realistic results on this dataset include:

- **AUROC (ROC curve area):** ~0.71 (Strong overall discrimination)
- **AUPRC (PR curve area):** ~0.05 (A classic indicator of extreme 2% minority class prevalence)
- **Recall (Sensitivity):** ~0.52 (Successfully catching over 52% of actual sepsis events among severe noise)
- **F2-Score:** ~0.17 (Prioritizing recall over precision)

## 📁 Repository Structure

* `Sepsyd_Model_Pipeline.ipynb`: The main End-to-End training pipeline. It handles loading the massive dataset, applying the complex temporal feature engineering, training the XGBoost model, and outputting the final metrics.
* `combined_sepsis_data.csv`: The core dataset containing 1,758,170 hours of clinical ICU data combined from 40,000+ individual patient files (Due to size limits, make sure this is available locally).

## ⚙️ How to Run

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas numpy xgboost scikit-learn
   ```
3. Make sure the `combined_sepsis_data.csv` file is placed in the same directory as the notebook.
4. Open the Jupyter Notebook:
   - `Sepsyd_Model_Pipeline.ipynb`
5. Click **"Restart & Run All"** to execute the pipeline. The notebook will automatically chunk the data, apply the 720-feature extraction, and evaluate the final XGBoost model.
