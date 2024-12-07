import pickle
from django.http import JsonResponse

# Load the trained model and vectorizer
with open(r'C:\Users\ilakk\Desktop\sentiment_analysis\sentiment\model_files\svc_model.pkl', 'rb') as model_file:
    svc_model = pickle.load(model_file)

with open(r'C:\Users\ilakk\Desktop\sentiment_analysis\sentiment\model_files\vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_sentiment(request):
    text = request.GET.get('text', '')  # Get text from URL parameter
    
    if text:
        # Transform the input text into features
        text_vectorized = vectorizer.transform([text])
        
        # Predict sentiment using the model
        prediction = svc_model.predict(text_vectorized)
        sentiment = 'positive' if prediction == 1 else 'negative'
        
        return JsonResponse({'sentiment': sentiment})
    else:
        return JsonResponse({'error': 'No text provided'}, status=400)
