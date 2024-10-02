import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained RNN model
model = tf.keras.models.load_model('My_model.keras', compile=False)


# Function to process and predict the input message
def make_prediction(message, model, tokenizer, max_length):
    # Tokenize and pad the input message
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    return prediction[0][0]

# Set the maximum length and vocabulary size for padding and tokenizing
max_length = 10
vocab_size = 10000  # Change this according to the tokenizer used during training

# Initialize the tokenizer (use the same tokenizer configuration as used in training)
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(["Example training data"])  # Placeholder; this should be replaced by actual training data

# Set page configuration for better UI
st.set_page_config(page_title="Spam Detection", page_icon="✉️", layout="centered")

# Add custom CSS to style the UI
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #1f77b4;
        }
        h2 {
            text-align: center;
            color: #ff7f0e;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #f0f2f6;
        }
    </style>
    """, unsafe_allow_html=True)

# Page title and subtitle
st.title("✉️ Spam Detection with RNN Model")
st.write("Enter a message to check if it's spam or not.")

# Text area for message input
input_message = st.text_area("Enter your message:", "")

# Submit button for classification
if st.button("Submit for Classification"):
    if input_message:
        st.write("Analyzing the message...")
        
        # Make prediction
        prediction = make_prediction(input_message, model, tokenizer, max_length)
        
        # Display the result with enhanced UI
        if prediction >= 0.5:
            st.markdown(
                "<h2 style='color: red;'>🚨 Spam Detected! 🚨</h2>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color: green;'>✅ No Spam Detected!</h2>", 
                unsafe_allow_html=True
            )
    else:
        st.write("Please enter a message.")

# Footer message
st.markdown(
    "<div class='footer'>Built with ❤️ using TensorFlow and Streamlit</div>", 
    unsafe_allow_html=True
)
