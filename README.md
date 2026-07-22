# Diabetes-Disease-Prediction 🩺

**Machine Learning Internship — CodeAlpha | Task 4: Disease Prediction from Medical Data**

A machine learning system that predicts the risk of **Diabetes** based on patient medical data, with a real-time interactive web GUI for live predictions on unseen data.

## 📊 Overview

This project uses the **Pima Indians Diabetes Dataset** (UCI Machine Learning Repository) to train and compare multiple classification algorithms, automatically selects the best-performing model, and serves it through a **Streamlit** web interface for real-time predictions.

- **Dataset**: Auto-downloaded from a public URL — no manual download needed
- **Algorithms compared**: Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost
- **Best model selected automatically** by 5-fold cross-validated ROC-AUC
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **GUI**: Real-time prediction on new/unseen patient data via a web form

## 🗂 Project Structure

```
CodeAlpha_DiseasePrediction/
├── train_model.py              # Downloads data, trains & evaluates models, saves best one
├── app.py                      # Streamlit GUI for real-time predictions
├── requirements.txt            # Python dependencies
├── model.pkl                   # Saved best model (generated after training)
├── scaler.pkl                  # Saved feature scaler (generated after training)
├── model_metadata.json         # Best model info + metrics (generated after training)
├── model_comparison_results.csv# Full comparison table of all models (generated)
└── README.md
```

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This downloads the dataset directly from a public URL, cleans it, trains 5 classification models, prints a full metrics comparison, and saves the best model to disk.

### 3. Launch the real-time prediction GUI
```bash
streamlit run app.py
```
This opens a browser window where you can enter a patient's medical values (Glucose, BMI, Age, etc.) and get an instant diabetes risk prediction with probability score.

## 📈 Results

The training script evaluates every model using Precision, Recall, F1-Score, and ROC-AUC (with 5-fold cross-validation), then automatically picks the model with the highest ROC-AUC. Full results are printed to console and saved in `model_comparison_results.csv`.

## 🧠 Approach

1. **Data preprocessing** — Biologically invalid zero-values (e.g. 0 blood pressure) are treated as missing and imputed with the column median.
2. **Feature scaling** — StandardScaler normalizes all features.
3. **Model comparison** — Multiple classifiers are trained and evaluated on a held-out test set.
4. **Model selection** — Best model chosen by ROC-AUC, then saved with `joblib`.
5. **Deployment** — Streamlit GUI loads the saved model and scaler for instant, real-time predictions on new patient data.

## ⚠️ Disclaimer

This tool is for educational/demonstration purposes only and does **not** constitute medical advice or diagnosis. Always consult a qualified healthcare professional.

## 👤 Author

**Raza Ahmad Khan**
CS & AI Student, UET Mardan
CodeAlpha Machine Learning Intern
