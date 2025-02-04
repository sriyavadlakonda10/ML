import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Streamlit app title
st.title("XGBoost Churn Prediction App")

# File uploader for dataset
uploaded_file = st.file_uploader(r"C:\Users\Sriya v\OneDrive\Desktop\Churn_Modelling.csv", type=["csv"])
if uploaded_file is not None:
    # Load the dataset from the uploaded file
    dataset = pd.read_csv(r"C:\Users\Sriya v\OneDrive\Desktop\Churn_Modelling.csv")
    st.write("Dataset Preview:")
    st.write(dataset.head())

    # Feature and target selection
    X = dataset.iloc[:, 3:-1].values  # Select features (columns 3 to second-to-last)
    y = dataset.iloc[:, -1].values    # Select target (last column)

    # Encoding categorical data
    # Label Encoding the "Gender" column
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])  # Assuming column index 2 is "Gender"

    # One Hot Encoding the "Geography" column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))  # Assuming column index 1 is "Geography"

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Training XGBoost on the Training set
    classifier = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.01, use_label_encoder=False, eval_metric='logloss')
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Displaying results
    st.subheader("Model Performance")
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    st.write("Accuracy Score:")
    ac = accuracy_score(y_test, y_pred)
    st.write(ac)

    st.write("Training Set Accuracy (Bias):")
    bias = classifier.score(X_train, y_train)
    st.write(bias)

    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')
    st.write(f"Cross-Validation Accuracy: {accuracies.mean() * 100:.2f} %")
    st.write(f"Cross-Validation Standard Deviation: {accuracies.std() * 100:.2f} %")
else:
    st.write(r"C:\Users\Sriya v\OneDrive\Desktop\Churn_Modelling.csv")