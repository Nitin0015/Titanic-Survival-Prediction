# **Titanic Survival Prediction**

This repository contains a Jupyter Notebook that demonstrates how to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset used in this project is the famous Titanic dataset.

---

## **Overview**

The sinking of the Titanic is one of the most infamous shipwrecks in history. Predicting which passengers survived this disaster based on their demographic and socioeconomic information is a classic machine learning problem. This project uses **Logistic Regression** to classify passengers as survivors (`1`) or non-survivors (`0`).

The dataset includes features such as passenger class, age, gender, and fare, with the target variable (`Survived`) indicating whether a passenger survived or not.

---

## **Dataset**

- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- **Features**:
  - `PassengerId`: Unique identifier for each passenger.
  - `Survived`: Target variable (1 = Survived, 0 = Did not survive).
  - `Pclass`: Passenger class (1st, 2nd, or 3rd).
  - `Name`: Name of the passenger.
  - `Sex`: Gender of the passenger.
  - `Age`: Age of the passenger.
  - `SibSp`: Number of siblings/spouses aboard.
  - `Parch`: Number of parents/children aboard.
  - `Ticket`: Ticket number.
  - `Fare`: Amount paid for the ticket.
  - `Cabin`: Cabin number (if available).
  - `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset (`train.csv`) is loaded into a Pandas DataFrame.
2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics and visualizations are generated using Seaborn and Matplotlib to explore relationships between features and survival rates.
3. **Data Preprocessing**:
   - Missing values are handled (e.g., imputing missing ages).
   - Categorical variables such as `Sex` and `Embarked` are encoded into numerical values for model training.
4. **Model Training**:
   - A Logistic Regression model is trained to predict survival outcomes.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Model Evaluation**:
   - Accuracy score is calculated to evaluate model performance.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/TitanicSurvivalPrediction.git
   cd TitanicSurvivalPrediction
   ```

2. Ensure that the dataset file (`train.csv`) is in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Titanic-Pred.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The Logistic Regression model provides predictions for passenger survival based on input features. The accuracy score indicates how well the model performs in predicting survival outcomes. Further improvements can be made by experimenting with feature engineering or using other machine learning models.

---

## **Acknowledgments**

- The dataset was sourced from [Kaggle](https://www.kaggle.com/c/titanic/data).
- Special thanks to Scikit-learn for providing robust machine learning tools.

---
