
# 🩺 Skin Disorder Prediction using Machine Learning (Multi-Class Classification)

---

## 1. 📌 Project Overview

Dermatologists often face challenges in differentiating **Erythemato-Squamous diseases** at early stages because many clinical symptoms overlap. Accurate diagnosis frequently requires invasive procedures like biopsies.

### Objective

To perform comprehensive Exploratory Data Analysis (EDA) and build high-accuracy Machine Learning classification models to predict skin diseases based on clinical symptoms, histopathological features, age, and family history.

### Goal



 ***Assist medical professionals in early and accurate diagnosis, reducing the dependency on invasive procedures and improving patient outcomes.***

---



## 📊 2. Data Understanding

The project utilizes a dermatology dataset with **35 attributes**:


* **34 Input Features**: Including 11 clinical attributes (e.g., erythema, scaling, itching) and 22 histopathological attributes (e.g., melanin incontinence, spongiosis).

* **Target Variable**: `class` representing 6 different skin diseases.

> ### Target Classes (Dermatology Diseases)


The dataset contains 6 classes representing:

1. Psoriasis
2. Seborrheic Dermatitis
3. Lichen Planus
4. Pityriasis Rosea
5. Chronic Dermatitis
6. Pityriasis Rubra Pilaris


---



## 📈 3. Exploratory Data Analysis (EDA)


### EDA included:

* **Age Distribution**: Most cases occur between 20–50 years, with a significant peak around age 35.
* **Symptom Severity**: Erythema (redness) is a primary symptom for the majority, with grade 2 being the most common severity level.
* **Class Balance**: The dataset shows imbalance, with Class 1 having the highest representation and Class 6 the lowest.

### Key Insights

* Dataset is relatively well-balanced across classes
* Strong correlations exist between certain histopathological features
* Age distribution varies across disease types
* Some classes show overlapping patterns (important later in misclassification)

---

## 🧹 4. Data Preprocessing

### 🔎 Handling Missing Values


* **Handling Missing Values**: Replaced "?" in the `Age` column with NaN and performed numerical conversion.
* **Scaling**: Implementation of `StandardScaler` and `MinMaxScaler` via Scikit-Learn pipelines to normalize feature ranges.
* **Pipeline Architecture**: Used `ColumnTransformer` and `Pipeline` for reproducible data transformation.
* **Age Column** : Treated `Age` separately from other dermatology features

---


## ⚙️ 5. Machine Learning Models Evaluated



The following models were evaluated using **Stratified K-Fold Cross Validation (5 folds):**


| Model                            |
| -----------------------------    |
|Support Vector Machine (SVM)      |
| Logistic Regression              |
| Random Forest                    | 
|K-Nearest Neighbors (KNN)         |
| * Gradient Boosting & XGBoost    |
| Decision Tree                    | 
---


## 🔁 6. Cross-Validation Strategy

To ensure the model generalizes well to unseen data and isn't biased by the specific ordering of the dataset, a robust validation strategy was implemented:

* **StratifiedKFold (n=5)** : Each of the 5 folds maintains the exact same percentage of each disease class as the original dataset.
* **Shuffle** = True
* **Random State** = 42
* **Metric**: Accuracy

Stratification ensures:

> Each fold maintains the same class distribution as the full dataset.

---

## 🏆 7. Model Comparison



After cross-validation:
### **1. Overall Model Performance Comparison**

This table compares the stability and general accuracy of each algorithm.

| Model | Mean Accuracy (%) | Std Deviation | Accuracy (Test Set) | Macro F1-Score |
| --- | --- | --- | --- | --- |
| **SVM** | **98.09%** | **0.0067** | **0.98** | **0.98** |
| **Logistic Regression** | 97.81% | 0.0110 | 0.98 | 0.98 |
| **KNN** | 97.54% | 0.0135 | 0.98 | 0.97 |
| **Random Forest** | 97.54% | 0.0160 | 0.98 | 0.97 |
| **Gradient Boosting** | 96.99% | 0.0055 | 0.97 | 0.97 |
| **Decision Tree** | 94.80% | 0.0280 | 0.95 | 0.94 |



### **2. Class-wise F1-Score Comparison**

Since dataset has 6 classes, this table shows which models performed best on specific categories. An F1-score of **1.00** indicates perfect classification for that class.

| Model | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 |
| --- | --- | --- | --- | --- | --- | --- |
| **Random Forest** | 0.99 | 0.93 | 1.00 | 0.93 | 1.00 | 0.97 |
| **Logistic Regression** | 1.00 | 0.94 | 1.00 | 0.92 | 1.00 | 1.00 |
| **SVM** | 1.00 | 0.94 | 1.00 | 0.93 | 1.00 | 1.00 |
| **KNN** | 1.00 | 0.92 | 1.00 | 0.91 | 1.00 | 1.00 |
| **Gradient Boosting** | 0.99 | 0.93 | 0.99 | 0.92 | 1.00 | 0.98 |
| **Decision Tree** | 0.98 | 0.89 | 0.98 | 0.90 | 0.95 | 0.92 |



🥇 **Best Performing Model: Support Vector Machine (SVM)**



SVM outperformed other models in terms of average cross-validation accuracy.

## 🥇 8. Best Performing Model

```**Support Vector Machine (SVM)**```

SVM appears to be your best model. It has the highest **Mean Accuracy (98.09%)** and the second-lowest Standard Deviation, meaning it is both highly accurate and very stable across different data folds.


Here is the hyperparameter tuning section formatted for a high-quality `README.md` file. I have organized it to highlight the methodology and the technical parameters used.

---

## 🔧 9. Hyperparameter Tuning

To optimize the performance of the **Support Vector Machine (SVM)**—identified as the top-performing model—we performed systematic hyperparameter optimization using **GridSearchCV**. This process ensures the model generalizes well to unseen data by finding the ideal balance between bias and variance.

### 🛠️ Tuning Methodology

The tuning process utilized the following robust validation techniques:

* **Grid Search:** An exhaustive search over a specified subset of the hyperparameter space.
* **Stratified K-Fold:** Ensures that each fold of the cross-validation maintains the same percentage of samples for each target class as the complete set.
* **Scoring Metric:** Optimized specifically for **Accuracy**, while monitoring F1-score to ensure balance across all 6 classes.

### 🔍 Hyperparameter Space

The following parameters were evaluated to find the optimal configuration:

| Parameter | Purpose | Values Tested |
| --- | --- | --- |
| **Kernel Type** | Determines the decision boundary shape | `linear`, `poly`, `rbf`, `sigmoid` |
| **C (Regularization)** | Controls the trade-off between smooth boundary and classifying training points correctly | `0.1`, `1`, `10`, `100` |
| **Gamma** | Defines how far the influence of a single training example reaches (for RBF/Poly) | `scale`, `auto`, `0.01`, `0.001` |



## 📈 10. Optimization Results


The grid search identified the optimal configuration for the SVM model, significantly stabilizing its predictive power across all classes.

* **Best Parameters:** `{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}`
* **Best Cross-Validated F1 Score:** 97.53%

#### **Final Model Performance**

The tuned model achieved an overall **accuracy of 0.98** on the test set. While classes 1, 3, 5, and 6 achieved perfect scores, the model shows a slight trade-off between Class 2 and Class 4:

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **Class 2** | 0.96 | 0.90 | 0.93 | 61 |
| **Class 4** | 0.89 | 0.96 | 0.92 | 49 |


## 📉 9. Confusion Matrix & Error Analysis

The following table represents the distribution of predictions across the 6 classes. Rows indicate the **Actual** classes, while columns indicate the **Predicted** labels.

| Actual \ Predicted | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
| --- | --- | --- | --- | --- | --- | --- |
| **Class 0** | **112** | 0 | 0 | 0 | 0 | 0 |
| **Class 1** | 0 | **55** | 0 | 6 | 0 | 0 |
| **Class 2** | 0 | 0 | **72** | 0 | 0 | 0 |
| **Class 3** | 0 | 2 | 0 | **47** | 0 | 0 |
| **Class 4** | 0 | 0 | 0 | 0 | **52** | 0 |
| **Class 5** | 0 | 0 | 0 | 0 | 0 | **20** |

###  Quick Summary

* **Total Samples:** 366
* **Correct Predictions:** 358 (Sum of the diagonal)
* **Incorrect Predictions:** 8 (Off-diagonal)
* **Primary Confusion:** Between **Class 1** and **Class 3**, which accounts for 100% of the model's errors.

## 🩺 10. Error Analysis
#### ***Class 2 = Seborrheic Dermatitis***
#### ***Class 4 = Pityriasis Rosea***

Clinically, these two can have:
* Similar scaling patterns
* Similar erythema values
* Similar lesion distribution

> The misclassifications occur only between class 2 and class 4, suggesting overlapping feature characteristics.

> The SVM model performs very well overall, but due to similarity in clinical attributes between these two classes, a few borderline samples fall on the opposite side of the hyperplane


---
## 💾 10. Model Saving

The final tuned SVM model was saved for deployment or future inference using serialization.
This ensures:

* Reproducibility
* Reusability
* Deployment readiness



---



## 🛠 Tech Stack


* **Language:** Python
* **Environment:** Jupyter Notebook
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (SVM, Random Forest, GridSearchCV)
* **Visualization:** Seaborn, Matplotlib



---



## 📂 Project Structure


```bash
DERMATOLOGY_DATASET_(Multi-class_classification)/
│
├── data/
│   └── dermatology.csv                # Original dermatology dataset
│
├── notebooks/
│   └── skin-disorder-prediction-multi-classification.ipynb
│                                       # Main ML workflow notebook
│
├── myenv/                              # Virtual environment (excluded via .gitignore)
│
├── .gitignore                          # Files ignored from version control
│
└── README.md                           # Project documentation
```

---




## 🚀 How to Run This Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Shravani-1325/Skin-Disorder-Prediction-ML--Multi-Classification-.git
cd Skin-Disorder-Prediction-ML--Multi-Classification-
```

### 2️⃣ Create & Activate Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Notebook

```bash
jupyter notebook
```

Open:

```
notebooks/skin-disorder-prediction-multi-classification.ipynb
```

Run all cells to reproduce results.

---



## 📊 Evaluation Metric

The model performance was evaluated using:

* **Accuracy** – Overall correctness of predictions
* **Precision** – Correct positive predictions per class
* **Recall** – Ability to correctly identify each class
* **F1-Score** – Balance between precision and recall
* **Stratified 5-Fold Cross-Validation** – Ensures stable and reliable performance across splits

Primary metric: **Accuracy (~98%)**

## 🙋🏻‍♀️Author
#### Shravani More

