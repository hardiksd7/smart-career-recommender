import streamlit as st
from PyPDF2 import PdfReader
import pickle
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Smart Career Path Recommender",
    page_icon="🧠",
    layout="centered",
)

# Title and instructions
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>Smart Career Path Recommender</h1>", unsafe_allow_html=True)
st.markdown("### 📝 Upload your Resume in PDF format to get personalized career suggestions based on your skills.")
st.markdown("---")

# Load model and vectorizer
with open("app/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("app/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Resume upload
uploaded_file = st.file_uploader("📄 Choose your resume (PDF only):", type=["pdf"])

if uploaded_file is not None:
    st.success("✅ Resume uploaded successfully!")

    # Read and extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    # Show extracted text (optional)
    with st.expander("📑 Show extracted resume text"):
        st.write(resume_text)

    # Predict button
    if st.button("🔍 Predict Career Path"):
        # Transform the text
        input_vector = vectorizer.transform([resume_text])

        # Make prediction
        prediction = model.predict(input_vector)[0]

        # Display result
        st.markdown(f"### 🎯 Recommended Career Path: **{prediction}**")

        # Additional message
        st.info("This recommendation is based on your resume content and the model's training data. Make sure your resume includes all your key skills and experiences.")
