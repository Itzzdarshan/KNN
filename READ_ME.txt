# 🚢 Ship Finder AI: Acoustic & Physical Vessel Classification

## 🚀 Business Problem
Identifying maritime vessels in high-traffic corridors is essential for naval safety and port management. This project implements a **K-Nearest Neighbors (KNN)** classification engine to categorize vessels based on sonar-derived and physical signatures, automating the identification of small personal craft, luxury liners, and industrial cargo carriers.

## 🧠 Technical Architecture
The system utilizes a proximity-based supervised learning pipeline to classify marine signatures into three distinct operational categories.

### 1. Feature Vector Composition
The model analyzes 5 key dimensions to establish a vessel's identity:
* **Physical Metrics:** `Length` and `People Capacity`.
* **Operational Metrics:** `Current Speed` and `Vessel Age`.
* **Acoustic Signature:** `Engine Noise Level` (Scale 1-10).

### 2. Engineering Decisions: KNN & Scaling
* **Algorithm Choice:** **K-Nearest Neighbors (k=5)**. 
    * *Justification:* KNN is highly effective for signature matching where similar vessel types exhibit clustered physical and acoustic properties.
* **Normalization Strategy:** Implemented `StandardScaler` (Z-score normalization).
    * *Critical Insight:* Since `People Capacity` (up to 5000) and `Engine Noise` (up to 10) exist on vastly different scales, normalization was required to ensure the distance-based KNN algorithm was not mathematically biased toward larger numeric features.


## 🛠️ Tech Stack & Tooling
* **Language:** Python 3.14
* **ML Engine:** Scikit-Learn (KNeighborsClassifier)
* **Data Science:** Pandas, NumPy
* **Interface:** Streamlit (Custom "Deep Sea" UI with Glassmorphism and CSS Injection)
* **Serialization:** Joblib (Persistence for Model and Scaling parameters)

## 📁 Repository Structure
```text
├── app.py                  # "Deep Sea" Radar Terminal (Streamlit Frontend)
├── train_model.py          # KNN Training pipeline with k-fold validation logic
├── model.pkl               # Serialized KNN classification object
├── scaler.pkl              # Serialized Z-score normalization parameters
└── vessel_signatures.csv    # Balanced dataset of 1,000 marine signatures