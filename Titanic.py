import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the prepared data
train_df = pd.read_csv('prepared_train.csv')
test_df = pd.read_csv('prepared_test.csv')

# Separate features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'FamilySize', 'IsAlone', 'FareBin', 'AgeBin', 'Deck']
X_train = train_df[features]
y_train = train_df['Survived']
X_test = test_df[features]
y_test = test_df['Survived']

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Average CV score:", cv_scores.mean())

# Make predictions
y_pred = rf_model.predict(X_test)
print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Survival Prediction')
plt.tight_layout()
plt.show()

# Survival analysis by key features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sex vs Survival
sns.barplot(data=train_df, x='Sex', y='Survived', ax=axes[0,0])
axes[0,0].set_title('Survival Rate by Sex')

# Pclass vs Survival
sns.barplot(data=train_df, x='Pclass', y='Survived', ax=axes[0,1])
axes[0,1].set_title('Survival Rate by Passenger Class')

# Age Groups vs Survival
sns.barplot(data=train_df, x='AgeBin', y='Survived', ax=axes[1,0])
axes[1,0].set_title('Survival Rate by Age Group')

# Family Size vs Survival
sns.barplot(data=train_df, x='FamilySize', y='Survived', ax=axes[1,1])
axes[1,1].set_title('Survival Rate by Family Size')

plt.tight_layout()
plt.show()
