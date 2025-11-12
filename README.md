# 🔧 Remaining Useful Life (RUL) Prediction Using LSTM

![Pipeline](results/pipeline_diagram.png)

Predicting the **Remaining Useful Life (RUL)** of machinery using sensor data and deep learning.  
This project focuses on **predictive maintenance** — identifying when a machine component is likely to fail,  
so it can be replaced **before breakdown** occurs.

---

## 🚀 Overview

This project implements a **predictive maintenance pipeline** using **time-series sensor data** from the NASA CMAPSS turbofan dataset (FD001).  
It leverages **Long Short-Term Memory (LSTM)** neural networks to predict the Remaining Useful Life of engines based on sensor readings over time.

### 🧩 Key Objectives:
- Process and clean raw sensor data.
- Generate sequences for temporal modeling.
- Train an **LSTM-based model** to predict RUL.
- Evaluate model performance and visualize degradation trends.

---

## 🗂️ Dataset

**Source:** NASA CMAPSS (FD001 subset)  
The dataset contains multi-sensor time-series data from turbofan engines under various operational conditions.  
Each engine runs until failure, allowing the model to learn degradation patterns.

| Feature | Description |
|----------|-------------|
| `id` | Engine ID |
| `cycle` | Operational cycle number |
| `sensor_1 ... sensor_21` | Sensor readings capturing engine behavior |
| `RUL` | Remaining Useful Life (target variable) |

---

## ⚙️ Approach

The workflow follows a systematic pipeline:

```text
Data Preprocessing → Sequence Generation → LSTM Training → RUL Prediction → Evaluation
```

### 🔄 Steps:
1. **Data Preprocessing**
   - Remove irrelevant sensors.
   - Normalize features using MinMax scaling.
   - Label RUL for each time step.

2. **Sequence Generation**
   - Convert continuous sensor readings into sequences for LSTM input.
   - Each sequence represents a time window of engine health.

3. **Model Training**
   - LSTM layers learn temporal dependencies in sensor data.
   - The network predicts remaining cycles until failure.

4. **Evaluation**
   - Metrics: RMSE, MAE, and R² Score.
   - Visualize predicted vs actual RUL values.

---

## 🧠 Model Architecture

```python
Model: "LSTM_RUL_Model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 100)               48800
dense_1 (Dense)              (None, 50)                5050
dense_2 (Dense)              (None, 1)                 51
=================================================================
Total params: 53,901
Trainable params: 53,901
_________________________________________________________________
```

---

## 📊 Results

### 📈 Predicted vs Actual RUL

![Results](results/model_performance.png)

- The LSTM model captures degradation patterns effectively.
- Predictions align closely with true RUL values, indicating robust temporal learning.

| Metric | Value |
|---------|-------|
| RMSE | 17.34 |
| MAE | 13.42 |
| R² | 0.91 |

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Model Deployment (Optional) | Streamlit / Flask |

---

## 🧪 How to Run

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/RUL-Prediction.git
cd RUL-Prediction

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the notebook
jupyter notebook RUL.ipynb
```

> Optional: Run the trained model directly with `FD001_RUL_LSTM_Model.h5`.

---

## 🔮 Future Work
- Extend to other CMAPSS subsets (FD002–FD004).
- Incorporate attention-based LSTMs or Transformers.
- Build a real-time monitoring dashboard using Streamlit.

---

## ✨ Author

**Pratik Shimpi**  
📧 [Your Email or LinkedIn link here]  
💡 *Predictive Maintenance using Deep Learning*

---

## 📁 Folder Structure
```
RUL-Prediction/
│
├── data/
│   └── train_FD001.csv
│
├── notebooks/
│   └── RUL.ipynb
│
├── results/
│   ├── pipeline_diagram.png
│   └── model_performance.png
│
├── FD001_RUL_LSTM_Model.h5
├── requirements.txt
└── README.md
```
---

⭐ **If you find this project useful, consider giving it a star!**
