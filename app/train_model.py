import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("C:/Users/Lenovo/Desktop/smart-career-recommender/data/resumes_dataset.csv")

# Clean skills (ensure consistent spacing)
df['skills'] = df['skills'].apply(lambda x: ' '.join(x.lower().split(',')))

# Features and labels
X = df['skills']
y = df['career']

# Convert text to vectors
cv = CountVectorizer()
X_vec = cv.fit_transform(X)

# Train a simple classifier
model = LogisticRegression()
model.fit(X_vec, y)

# Save the model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved as model.pkl & vectorizer.pkl")
