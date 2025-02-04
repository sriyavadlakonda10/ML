import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import streamlit as st

# Load dataset
@st.cache_data  # Use st.cache_data for caching data
def load_data():
    dataset = pd.read_csv(r"C:\Users\Sriya v\OneDrive\Desktop\Social_Network_Ads.csv")
    return dataset

dataset = load_data()

# Split dataset into independent and dependent variables
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling (Apply only after splitting the data)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)  # Fit & transform training set
x_test = sc.transform(x_test)  # Transform test set

# Train the SVM classifier
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

# Predict the test set results
y_pred = classifier.predict(x_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(cm)

# Cross-validation accuracy
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
st.write("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
st.write("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))

# Grid Search for best parameters
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
st.write("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
st.write("Best Parameters:", best_parameters)

# Plotting the decision boundary
def plot_decision_boundary(X_set, y_set, title):
    fig, ax = plt.subplots()  # Create a figure and axis
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    ax.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
    ax.set_xlim(X1.min(), X1.max())
    ax.set_ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                   color=ListedColormap(('red', 'green'))(i), label=j)
    ax.set_title(title)
    ax.set_xlabel('Age')
    ax.set_ylabel('Estimated Salary')
    ax.legend()
    st.pyplot(fig)  # Pass the figure to st.pyplot()

# Display training set results
st.write("Training Set Results:")
plot_decision_boundary(x_train, y_train, 'Kernel SVM (Training set)')

# Display test set results
st.write("Test Set Results:")
plot_decision_boundary(x_test, y_test, 'Kernel SVM (Test set)')