# Predictive Maintenance for Industrial Pneumatic Actuator (DAMADICS Benchmark)

## Overview
This repository implements a Machine Learning and signal processing pipeline for early fault detection and predictive maintenance on an industrial pneumatic actuator, evaluated on the DAMADICS benchmark.

The goal is to identify mechanical drift (friction and internal leakage) up to 48 hours prior to catastrophic failure.

---

## Key Results
- Overall Accuracy: 91% on test set
- Alert Precision: 100% (zero critical false alarms)
- Lead-time Horizon: 48 hours anticipation before actuator blockage
- Key Feature: Position tracking error and rolling statistics account for over 70% of feature importance

---

## Methodology
1. Data loading from multivariate sensor records (consigne, position, pression, debit).
2. Physics-informed feature engineering (algebraic residual, derivatives, rolling mean/std).
3. Supervised classification using Random Forest (120 estimators).
4. Performance evaluation (Confusion Matrix, Precision/Recall, Feature Importance).

---

## How to Run
1. Install requirements:
   pip install pandas numpy scikit-learn matplotlib seaborn
2. Run the script:
   python damadics_fault_detection.py
