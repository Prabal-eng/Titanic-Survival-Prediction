# Titanic Survival Prediction with Random Forest

This repository contains a machine learning project to predict passenger survival on the Titanic using a **Random Forest Classifier**. The model uses key demographic and socio-economic features to predict survival, including passenger class, age, sex, family size, and fare information.

## Project Overview

The goal of this project is to analyze and predict the likelihood of survival for Titanic passengers based on various features using:
- **Random Forest Classifier**: A powerful ensemble algorithm.
- **Cross-validation**: For robust model evaluation.
- **Feature Importance Analysis**: To understand which features contribute most to predictions.
- **Data Visualization**: Survival rates by features like age, sex, and family size.

## Dataset
The dataset has been preprocessed into `prepared_train.csv` and `prepared_test.csv` files, including engineered features:
- **Pclass**: Passenger class (1, 2, or 3)
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Fare paid for ticket
- **Embarked**: Port of embarkation
- **Title**: Passenger’s title (e.g., Mr., Mrs., Miss, etc.)
- **FamilySize**: Total family members aboard
- **IsAlone**: Binary indicator for solo travelers
- **FareBin**: Binned fare categories
- **AgeBin**: Binned age categories
- **Deck**: Cabin deck

## Installation
1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```
2. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Add the data files**:
   Place `prepared_train.csv` and `prepared_test.csv` files in the root folder.

## Usage
1. **Train and Evaluate Model**: 
   Run the code in `titanic_survival_prediction.py` to train the model and evaluate performance.

2. **Outputs**:
   - **Cross-validation scores**: Displays model accuracy across different folds.
   - **Test Set Accuracy** and **Classification Report**: Reports on test set performance.
   - **Feature Importance Plot**: Displays feature contributions to predictions.
   - **Survival Analysis**: Visualizes survival rates by key features (e.g., sex, age group, family size).

## Code Overview

- **Feature Engineering**: Uses demographic, socio-economic, and engineered features to enhance prediction accuracy.
- **Random Forest Model**: The classifier is trained on the training set and evaluated with 5-fold cross-validation.
- **Feature Importance**: Important features for survival prediction are visualized in a bar plot.
- **Exploratory Data Analysis (EDA)**: Visualizations show survival rates across various features, helping understand patterns in the data.

## Visualization Samples

1. **Feature Importance**: Displays which features impact survival predictions the most.
2. **Survival by Key Features**: Visualizes survival rates by sex, class, age group, and family size to provide insights.

## Example Commands
- **Training and Prediction**:
  ```python
  rf_model.fit(X_train, y_train)
  y_pred = rf_model.predict(X_test)
  ```
- **Cross-validation**:
  ```python
  cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
  ```
- **Plotting Feature Importance**:
  ```python
  sns.barplot(x='importance', y='feature', data=feature_importance)
  ```

## Future Work
- **Hyperparameter Tuning**: Further improve accuracy by optimizing the Random Forest model parameters.
- **Additional Feature Engineering**: Explore additional features such as ticket prefixes and family connections.
- **Testing Other Models**: Experiment with other classifiers like Gradient Boosting, SVM, or XGBoost for comparison.
