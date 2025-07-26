import streamlit as st
import os
import re
import pickle
from PyPDF2 import PdfReader

# Load ML model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Skill list
SKILL_SET = [
    "python", "java", "c++", "machine learning", "deep learning", "nlp", "sql", "mongodb",
    "cassandra", "linux", "aws", "azure", "cloud", "flask", "django", "excel", "tableau",
    "power bi", "hadoop", "spark", "git", "github", "r", "matplotlib", "seaborn", "numpy",
    "pandas", "data analysis", "data visualization", "opencv", "html", "css", "javascript",
    "react", "kafka", "airflow", "statistics", "probability", "data cleaning"
]

# Page setup
st.set_page_config(page_title="Smart Career Recommender", layout="wide")

# Header
st.markdown("<h1 style='color:navy;'>🚀 Smart Career Recommender</h1>", unsafe_allow_html=True)
st.markdown("Upload your resume and get a **predicted career path** using Machine Learning.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
This is a B.Tech CSE Final Year Project built by **Hardik Sharma**.
- Resume Parsing
- Skill Extraction
- ML-based Career Prediction
- Deployable Streamlit App
""")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your resume (PDF only)", type=["pdf"])

if uploaded_file:
    # Save and read PDF
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    pdf_reader = PdfReader("temp_resume.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Extract email
    email_match = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match[0] if email_match else "Not found"

    # Skill extraction
    extracted_skills = [skill for skill in SKILL_SET if skill in text.lower()]
    
    # Vectorize & predict
    vector = vectorizer.transform([" ".join(extracted_skills)])
    prediction = model.predict(vector)[0]

    # Display
    st.success("✅ Resume processed successfully!")

    st.markdown(f"### 📧 Email: `{email}`")
    st.markdown("### 🛠️ Extracted Skills:")
    st.markdown(", ".join(f"`{skill}`" for skill in extracted_skills))

    st.markdown(f"### 🎯 **Recommended Career Path:**")
    st.markdown(f"<h2 style='color:green;'>{prediction}</h2>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("💡 *Built using Python, Streamlit, ML & NLP.*")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Hardik Sharma | [GitHub](https://github.com/hardiksd7)")

