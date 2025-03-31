from django.shortcuts import render
import numpy as np
import re
import os
from decouple import config 
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from django.core.files.storage import FileSystemStorage
from gensim.models import KeyedVectors
from keras.models import load_model
import google.generativeai as genai

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load AI models
lstm_model1 = load_model("grading/models/essay_rank_lstm_1.keras")
lstm_model2 = load_model("grading/models/essay_rank_lstm_2.keras")
word2vec_model = KeyedVectors.load("grading/models/word2vecmodel.bin", mmap="r")


# Load API Key securely
GEMINI_API_KEY = config("GEMINI_API_KEY")  # Fetch from .env file

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")


def perform_ocr(image_path):
    """Extract text from an image using OCR."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def essay_to_vector(essay, model):
    """Convert an essay into a numerical vector using a pre-trained word2vec model."""
    stop_words = set(stopwords.words("english"))
    essay = re.sub("[^A-Za-z]", " ", essay).lower()
    words = word_tokenize(essay)
    words = [w for w in words if w not in stop_words]

    essay_vec = np.zeros((model.vector_size,), dtype="float32")
    no_of_words = 0

    for word in words:
        if word in model:
            no_of_words += 1
            essay_vec = np.add(essay_vec, model[word])

    if no_of_words != 0:
        essay_vec = np.divide(essay_vec, no_of_words)

    return essay_vec


def reshape_for_lstm(vector):
    """Reshape vector for LSTM model input."""
    return np.reshape(vector, (1, 1, -1))



def predict_score(essay):
    """Predict the score of an essay using LSTM models."""
    vector = essay_to_vector(essay, word2vec_model)
    vector = reshape_for_lstm(vector)
    prediction = (lstm_model1.predict(vector) + lstm_model2.predict(vector)) / 2
    score = np.argmax(prediction)

    # Apply bias correction
    bias = 2
    adjusted_score = min(score + bias, 10)
    return adjusted_score


def home(request):
    """Handle essay submission and generate AI explanation."""
    if request.method == "POST":
        user_essay = request.POST.get("essay", "")
        uploaded_file = request.FILES.get("image", None)

        # Extract text from uploaded image (if any)
        if uploaded_file:
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            extracted_text = perform_ocr(fs.path(filename))
        else:
            extracted_text = user_essay

        if extracted_text:
            predicted_score = predict_score(extracted_text)

            # Updated AI prompt for structured explanation
            prompt = f"""
            Analyze the following essay and provide structured feedback:
            Essay: "{extracted_text}"
            Given Score: {predicted_score}/10
            
            Response Format:
            **Strengths:** 
            - [Provide key strengths here]

            **Weaknesses:** 
            - [Provide key weaknesses here]

            **Suggested Improvements:** 
            - [Provide suggestions for improvement here]
            """
            response = gemini_model.generate_content(prompt)
            explanation_text = response.text

            # Extract Strengths, Weaknesses, and Improvements
            strengths, weaknesses, improvements = "", "", ""
            sections = re.split(r'\*\*Strengths:\*\*|\*\*Weaknesses:\*\*|\*\*Suggested Improvements:\*\*', explanation_text)

            if len(sections) > 1:
                strengths = sections[1].strip() if len(sections) > 1 else ""
                weaknesses = sections[2].strip() if len(sections) > 2 else ""
                improvements = sections[3].strip() if len(sections) > 3 else ""

            return render(
                request,
                "grading/result.html",
                {
                    "score": predicted_score,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "improvements": improvements,
                },
            )

    return render(request, "grading/home.html")
