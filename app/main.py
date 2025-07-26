from resume_parser import parse_resume
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_career(skills_list):
    # Join list into string
    skills_text = " ".join(skills_list)
    # Transform using vectorizer
    vectorized = vectorizer.transform([skills_text])
    # Predict
    prediction = model.predict(vectorized)
    return prediction[0]

if __name__ == "__main__":
    resume_path = "../resume_samples/sample_resume1.pdf"
    
    # Step 1: Parse Resume
    parsed = parse_resume(resume_path)
    print("📄 Resume Parsed Successfully")
    print("Name:", parsed['name'])
    print("Email:", parsed['email'])
    print("Skills:", parsed['skills'])

    # Step 2: Predict Career
    predicted_career = predict_career(parsed['skills'])
    print("🎯 Predicted Career Path:", predicted_career)
