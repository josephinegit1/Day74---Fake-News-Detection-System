from flask import Flask, request, render_template
import pickle
import string
import re

# --- Load Model and Vectorizer ---
model = pickle.load(open('best_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# --- Text Cleaning Function (same as training) ---
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Home Page ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Predict Route ---
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        cleaned_input = clean_text(news_text)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        return render_template('index.html', prediction=result)

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
