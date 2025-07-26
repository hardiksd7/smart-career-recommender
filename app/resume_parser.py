import fitz  # PyMuPDF
import spacy
import re

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Skill keywords (expand this list as needed)
SKILLS_DB = ["python", "java", "sql", "excel", "pandas", "numpy", "html", "css", 
             "javascript", "c++", "machine learning", "data analysis", "deep learning"]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_email(text):
    match = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match[0] if match else None

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_skills(text):
    text = text.lower()
    extracted = []
    for skill in SKILLS_DB:
        if skill in text:
            extracted.append(skill)
    return list(set(extracted))  # Remove duplicates

def parse_resume(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    name = extract_name(text)
    email = extract_email(text)
    skills = extract_skills(text)
    
    return {
        "name": name,
        "email": email,
        "skills": skills,
        "raw_text": text[:300]  # optional: preview
    }
