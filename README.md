
---

# Credit Card Fraud Detection Using Deep Learning

Detect and classify fraudulent credit card transactions using deep learning techniques.

This project builds a deep learning based model to identify credit card fraud using historical transaction data. Fraud detection is challenging due to the extremely **imbalanced dataset**, where only a very small percentage of transactions are fraudulent. ([Kaggle][1])

---

##  Overview

Fraudulent financial transactions cause huge monetary loss to individuals and organizations. Machine learning and deep learning techniques can be used to analyze historical transaction records to distinguish between **legitimate** and **fraudulent** activities.

This repository contains a Jupyter Notebook (`Detecting_Credit_Card_Fraud.ipynb`) that:

* Loads and preprocesses the transaction dataset
* Handles class imbalance
* Builds and evaluates a deep learning model
* Visualizes performance metrics

---

##  Dataset

You’ll need to download the dataset to run the notebook:

 **Credit Card Fraud Dataset** (from Kaggle):
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

This dataset contains **284,807 transactions**, of which **492 are fraudulent**. It has been anonymized using PCA transformations on the features. ([Kaggle][1])

After downloading, place the `creditcard.csv` in the root of the repository or update the path in the notebook accordingly.

---

##  Getting Started

### Prerequisites

Make sure you have the following installed:

* Python 3.7+
* Jupyter Notebook / Jupyter Lab
* Virtual environment (recommended)

###  Install Dependencies

You can install all required libraries using:

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## Run the Notebook

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

##  What’s Inside the Notebook

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


## 📊 Results & Evaluation

Model performance is evaluated using standard classification metrics. Since credit card fraud detection is a **highly imbalanced classification problem**, metrics such as **Precision, Recall, and F1-Score** are considered more important than accuracy alone.

Both **Machine Learning (ML)** and **Deep Learning (DL)** models were implemented and compared to analyze their effectiveness in fraud detection.

###  Models Used

**Machine Learning Models**

* Logistic Regression
* Random Forest Classifier,
* Linear SVC
* Gradient Boosting

**Deep Learning Models**

* Recurrent Neural Network (RNN)
* Long Short-Term Memory (LSTM)
* Shallow Neural Network (NN)
* 

---

###  Performance Comparison

The performance of ML and DL models is compared using evaluation metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score

A **bar plot** is used to visualize the comparison between ML and DL models, highlighting the superiority of deep learning models in capturing sequential patterns and improving fraud detection performance.

 *Bar plot comparing ML vs DL model performance is shown below:*

<img width="1205" height="666" alt="image" src="https://github.com/user-attachments/assets/a8e33665-dd41-49a6-a2d8-9f3da7b3b513" />


---

###  Deep Learning Model Results

####  RNN Model
* Architechture:
  <img width="1073" height="642" alt="image" src="https://github.com/user-attachments/assets/476d66eb-596d-4f8e-9f18-1d9a0054e4a8" />
* Accuracy:99.6%

* Evaluation metrics:
 <img width="627" height="208" alt="image" src="https://github.com/user-attachments/assets/f9d7db9f-e2f0-46d5-9b85-989e27917828" />
 
* RNN Model Loss vs Accuracy Graph:
<img width="1223" height="503" alt="image" src="https://github.com/user-attachments/assets/8ef98e7e-aee1-4448-b7db-de893d189db2" />



#### 🔹 LSTM Model
*Architecture:
<img width="1052" height="398" alt="image" src="https://github.com/user-attachments/assets/19893249-dfdc-43b1-9983-4587092523e3" />

* Accuracy: 
<img width="1183" height="455" alt="image" src="https://github.com/user-attachments/assets/c5d11e14-ab5d-4f11-8ba0-df63b9c40084" />

* LSTM Model Loss vs Accuracy graph:
  <img width="1189" height="499" alt="image" src="https://github.com/user-attachments/assets/cbb67bbb-95ba-4b3b-b485-4c7e8ca18076" />


---

###  Key Observations

* Deep learning models (**RNN and LSTM**) outperform traditional ML models in detecting fraudulent transactions.
* **LSTM** achieves better recall and F1-score due to its ability to capture long-term dependencies in transaction sequences.
* ML models perform reasonably well but struggle with minority class detection compared to DL models.

---

### Conclusion

Among all the models tested, **LSTM demonstrates the best overall performance**, making it the most suitable model for credit card fraud detection in this project.






