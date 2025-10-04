import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filepath):
    """
    Load the churn dataset from a CSV file.
    """
    data = pd.read_csv(filepath)
    return data

def perform_eda(data):
    """
    Perform exploratory data analysis on the dataset.
    """
    print("Data Info:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Plot distribution of target variable
    plt.figure(figsize=(6,4))
    sns.countplot(x='Exited', data=data)
    plt.title('Distribution of Target Variable (Exited)')
    # plt.show()  # Commented out for terminal execution

    # Plot correlation heatmap for numerical features
    plt.figure(figsize=(12,8))
    numerical_data = data.select_dtypes(include=[np.number])
    corr = numerical_data.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    # plt.show()  # Commented out for terminal execution

def preprocess_data(data):
    """
    Preprocess the data: encode categoricals, scale features, split into train/test.
    """
    # Drop unnecessary columns
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

    # Separate features and target
    X = data.drop('Exited', axis=1)
    y = data['Exited']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier on the training data.
    """
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

if __name__ == "__main__":
    data = load_data("Churn_Modelling.csv")
    print("Data loaded successfully. Here are the first 5 rows:")
    print(data.head())
    perform_eda(data)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print("Data preprocessing completed. Shapes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    clf = train_random_forest(X_train, y_train)
    print("Random Forest model trained successfully.")

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
