# Autoregressive Time Series Forecasting

## 📘 Overview

This project implements an **autoregressive forecasting model** for time series data using a hybrid architecture combining **Bidirectional GRUs** and **1D Convolutional layers**. The trained model performs **multi-step autoregressive forecasting**, where it iteratively predicts small windows (3 steps at a time) to cover the full forecasting horizon (9 steps total).

---

## ✨ Features

- Preprocessing and cleaning of irregular-length time series
- Sliding window segmentation with configurable parameters
- Label-aware train/validation/test split using stratification
- Deep learning model combining GRU layers and Conv1D layers
- Multi-step prediction using an autoregressive loop
- Model training with early stopping and learning rate scheduling
- Visualizations of training loss and learning rate evolution

---

## 📂 Dataset

The time series data is loaded from `.npy` files:

- `training_data.npy`: raw sequences  
- `categories.npy`: class labels  
- `valid_periods.npy`: valid start/end indices per sample  

The sequences are trimmed and padded to ensure minimum lengths and uniform windowing.

---

## 🚀 Usage

1. Mount your Google Drive in Colab
2. Place your `.npy` data files in the `Dataset` folder
3. Run the notebook to:
   - Preprocess data  
   - Build sequences  
   - Train the model  
   - Perform autoregressive forecasting  
   - Save and reload the trained model  

---

## 📤 Output

- 📉 Training/validation loss plot with best epoch marker  
- 📉 Learning rate schedule visualization  
- 💾 Final model saved to disk:

---

## 🧰 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn

---

## 📄 License

This project is intended for **academic and research purposes only**.  
You may adapt it as needed for your dataset or use case.
