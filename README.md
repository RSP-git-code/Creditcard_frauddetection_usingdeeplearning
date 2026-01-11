
---

# 🕵️‍♂️ Credit Card Fraud Detection Using Deep Learning

Detect and classify fraudulent credit card transactions using deep learning techniques.

This project builds a deep learning based model to identify credit card fraud using historical transaction data. Fraud detection is challenging due to the extremely **imbalanced dataset**, where only a very small percentage of transactions are fraudulent. ([Kaggle][1])

---

## 📌 Overview

Fraudulent financial transactions cause huge monetary loss to individuals and organizations. Machine learning and deep learning techniques can be used to analyze historical transaction records to distinguish between **legitimate** and **fraudulent** activities.

This repository contains a Jupyter Notebook (`Detecting_Credit_Card_Fraud.ipynb`) that:

* Loads and preprocesses the transaction dataset
* Handles class imbalance
* Builds and evaluates a deep learning model
* Visualizes performance metrics

---

## 📁 Dataset

You’ll need to download the dataset to run the notebook:

📥 **Credit Card Fraud Dataset** (from Kaggle):
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

This dataset contains **284,807 transactions**, of which **492 are fraudulent**. It has been anonymized using PCA transformations on the features. ([Kaggle][1])

After downloading, place the `creditcard.csv` in the root of the repository or update the path in the notebook accordingly.

---

## 🚀 Getting Started

### 🛠 Prerequisites

Make sure you have the following installed:

* Python 3.7+
* Jupyter Notebook / Jupyter Lab
* Virtual environment (recommended)

### 📦 Install Dependencies

You can install all required libraries using:

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## 📌 Run the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/RSP-git-code/Creditcard_frauddetection_usingdeeplearning.git
   cd Creditcard_frauddetection_usingdeeplearning
   ```

2. Place the **creditcard.csv** dataset in the project folder.

3. Open the notebook:

   ```bash
   jupyter notebook Detecting_Credit_Card_Fraud.ipynb
   ```

4. Run all cells in sequence.

---

## 📊 What’s Inside the Notebook

The notebook includes:

* **Data Loading & Inspection**
  Load the credit card transaction dataset and inspect its structure.

* **Exploratory Data Analysis (EDA)**
  Visualize distributions of features and class imbalance.

* **Preprocessing**
  Scale features and prepare train/test splits.

* **Handling Imbalanced Data**
  Use techniques such as oversampling or SMOTE to deal with class imbalance (common step in fraud detection workflows).

* **Deep Learning Model**
  Build a neural network classifier (e.g., using TensorFlow/Keras) to detect fraud.

* **Evaluation**
  Evaluate model performance with metrics such as:

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, F1-Score

---

## 📈 Results & Evaluation

Model performance is evaluated using standard classification metrics. Because fraud detection is highly imbalanced, **precision and recall** are more important than accuracy alone.
 ML vs DL model Perfomance in credit card fraud detection is visualized in barplot as shown below:
 

---






