# Autism Screening & Risk Prediction System

A machine learning-based command-line application that screens for potential Autism Spectrum Disorder (ASD) traits using behavioral questionnaire responses and demographic information.

> **Disclaimer:** This project is intended for educational and screening purposes only. It is not a medical or clinical diagnostic tool.

---

## 📌 Project Overview

The Autism Screening & Risk Prediction System uses a **Random Forest Classifier** to predict potential ASD traits based on questionnaire responses and user information.

The system takes behavioral assessment scores along with demographic and family-history information, processes the data, trains a machine learning model, and provides an interactive prediction with a confidence score and risk category.

---

## 🚀 Features

- Machine learning-based ASD screening
- Data cleaning and preprocessing
- Missing-value handling
- Categorical data encoding
- Feature selection
- Train/test data splitting
- Feature scaling using `StandardScaler`
- Random Forest classification
- Interactive command-line questionnaire
- Prediction confidence score
- Low, Medium, and High risk categorization
- Input validation and error handling

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Random Forest Classifier**
- **StandardScaler**
- **Google Colab / Jupyter Notebook**
- **GitHub**

---

## 📂 Project Structure

```text
Autism-Screening/
│
├── autism_screening.py
├── autism_data.csv
└── README.md
