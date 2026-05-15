# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Karthi V
RegisterNumber:  212225230130
*/
import pandas as pd

# Load dataset
data = pd.read_csv("Employee.csv")

# Display first rows
print("data.head():")
print(data.head())

# Dataset information
print("\ndata.info():")
print(data.info())

# Null values
print("\nisnull() and sum():")
print(data.isnull().sum())

# Value counts of target column
print("\ndata value counts():")
print(data["left"].value_counts())

# Label Encoding for salary column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

print("\nEncoding Salary Column:")
data["salary"] = le.fit_transform(data["salary"])

print(data.head())

# Feature selection
print("\nx.head():")

x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]

print(x.head())

# Target column
y = data["left"]

# Split dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy")

# Train model
dt.fit(x_train, y_train)

# Predictions
y_pred = dt.predict(x_test)

# Accuracy
print("\nAccuracy value:")

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)

# Sample Prediction
print("\nData Prediction:")

prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

print(prediction)

# Plot Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    dt,
    feature_names=x.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)

plt.show()
```

## Output:
<img width="1383" height="722" alt="Screenshot 2026-05-15 140744" src="https://github.com/user-attachments/assets/9ed4f208-d678-433e-a4f7-b48ad2074f8d" />
<img width="632" height="672" alt="Screenshot 2026-05-15 140718" src="https://github.com/user-attachments/assets/44d18a42-f3c1-4f79-933d-326e074a3383" />
<img width="263" height="322" alt="Screenshot 2026-05-15 140709" src="https://github.com/user-attachments/assets/e673188d-ef01-48f2-977f-ce8d12210b75" />
<img width="642" height="702" alt="Screenshot 2026-05-15 140641" src="https://github.com/user-attachments/assets/f82f561a-89a2-4bab-99c4-1ed2a09afed9" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
