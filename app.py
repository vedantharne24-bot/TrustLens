from flask import Flask, render_template, request, jsonify
import pickle
import requests
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the single Pipeline object
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def scrape_all_reviews(url):
    """Scrapes multiple reviews from common HTML structures."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for common review-related classes
        potential_tags = soup.find_all(['p', 'div', 'span'], 
            class_=re.compile(r'review-text|content|comment-body', re.I))
        
        # Extract unique texts longer than 40 chars
        reviews = list(set([r.get_text(strip=True) for r in potential_tags if len(r.get_text()) > 40]))
        return reviews if reviews else None
    except Exception as e:
        print(f"Scraping Error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = data.get('text', '')
        is_link = data.get('is_link', False)
        final_results = []

        if is_link:
            review_list = scrape_all_reviews(input_data)
            if not review_list:
                return jsonify({'error': 'Neural Link severed: No reviews found on this URL.'})
            texts_to_analyze = review_list[:8] # Limit for performance
        else:
            texts_to_analyze = [input_data]

        for text in texts_to_analyze:
            prediction = model.predict([text])[0]
            probs = model.predict_proba([text])[0]
            final_results.append({
                'text': text[:140] + "...",
                'prediction': "Fake" if prediction == 1 else "Real",
                'confidence': round(max(probs) * 100, 2)
            })

        return jsonify({'results': final_results})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)