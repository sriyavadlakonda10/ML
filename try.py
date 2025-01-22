import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Streamlit app
def main():
    st.title("Purchase Prediction App")
    st.subheader("Enter Age and Estimated Salary to predict if the product is purchased")

    # Input fields for Age and Estimated Salary
    age = st.number_input("Age", min_value=0, max_value=100, step=1, value=25)
    estimated_salary = st.number_input("Estimated Salary", min_value=0, step=1000, value=50000)

    # Predict button
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data = np.array([[age, estimated_salary]])
        
        # Make prediction
        prediction = classifier.predict(input_data)

        # Display the result
        if prediction[0] == 0:
            st.error("Prediction: Not Purchased")
        else:
            st.success("Prediction: Purchased")

if __name__ == "__main__":
    main()
