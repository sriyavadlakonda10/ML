import pickle
import numpy as np
import streamlit as st
import os

# Define the path to the model file
model_path = r"C:\Users\Sriya v\VS CODE\machine learning\Capstone projects\height&weight\final_model.pkl"

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found. Please ensure the file is in the correct directory.")
else:
    # Load the saved model
    try:
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Custom CSS for colorful representation
    st.markdown(
        """
        <style>
        .title {
            color: #FF5733;
            text-align: center;
            font-size: 32px;
        }
        .text {
            color: #7D3C98;
            text-align: center;
            font-size: 18px;
        }
        .prediction {
            color: #6C3483;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the Streamlit web app
    st.markdown('<p class="title">Weight Prediction App</p>', unsafe_allow_html=True)
    st.markdown('<p class="text">Enter your height in feet to predict your weight.</p>', unsafe_allow_html=True)

    # Default value for height
    default_height = 5.8

    # Input height from the user
    height_input = st.number_input("Enter the height in feet:", value=default_height, min_value=0.0)

    # Predict button
    if st.button('Predict'):
        try:
            # Reshape the input height to match the shape expected by the model (2D array)
            height_input_2d = np.array([[height_input]])

            # Use the loaded model to make predictions
            predicted_weight = loaded_model.predict(height_input_2d)

            # Convert predicted_weight to a scalar for display
            predicted_weight_value = float(predicted_weight[0])

            # Display the predicted weight
            st.markdown(
                f'<p class="prediction">Predicted weight: {predicted_weight_value:.2f} kg</p>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    