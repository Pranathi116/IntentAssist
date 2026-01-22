import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading dataset...")

# Load dataset
data = pd.read_csv("intent_data.csv")

texts = data["text"]
labels = data["intent"]

print("Dataset loaded successfully!")
print("Total samples:", len(data))

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print("Data split into training and testing sets")

# Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Text vectorization completed")

# Train the ML model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Model training completed")

# Test the model
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")

