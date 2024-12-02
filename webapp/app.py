import streamlit as st
import pdfplumber
import os
import google.generativeai as genai
from difflib import unified_diff
from dotenv import load_dotenv
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu

# Load environment variables
load_dotenv()

# Configure Google Gemini-Pro AI Model
genai.configure(api_key=os.getenv("AIzaSyDjqgQcMvCBqA8A9CrfQxUEn4rRu5ooPcM"))
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8129,
}

# Safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# Initialize the Google Gemini Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Streamlit page configuration
st.set_page_config(page_title="Radiology Report Analysis", page_icon="🧬", layout="wide")
st.title("Visualize Radiology Report and Correction 🌡")
st.subheader("An App to help with Radiology Analysis using AI")

# Define placeholders for generated_report and ground_truth
generated_report = "Patient has a normal chest X-ray. No abnormalities detected."
ground_truth = "Patient has a normal chest X-ray. No fractures or lesions."

# Function to extract text from uploaded file
def extract_text_from_file(file):
    if file.type == "application/pdf":
        try:
            with pdfplumber.open(file) as pdf:
                return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return None
    elif file.type == "text/plain":
        try:
            return file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return None
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        return None

# Generate a plain text comparison
def generate_comparison_text(generated_report, ground_truth):
    diff = unified_diff(
        ground_truth.splitlines(), 
        generated_report.splitlines(), 
        fromfile="Ground Truth", 
        tofile="Generated Report", 
        lineterm=""
    )
    return "\n".join(diff)

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Improved error detection
def find_errors(generated, ground_truth):
    gen_doc = nlp(generated.lower())
    truth_doc = nlp(ground_truth.lower())

    missed_entities = [(ent.text, "Missed") for ent in truth_doc.ents if ent.text.lower() not in generated.lower()]
    extra_entities = [(ent.text, "Extra") for ent in gen_doc.ents if ent.text.lower() not in ground_truth.lower()]

    return missed_entities + extra_entities

# Identify errors
errors = find_errors(generated_report, ground_truth)

# METRICS CALCULATION
def calculate_bleu(generated, ground_truth):
    return sentence_bleu([ground_truth.split()], generated.split())

# Streamlit UI for uploading file and viewing outputs
uploaded_file = st.file_uploader("Upload a Radiology Report (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    report_text = extract_text_from_file(uploaded_file)
    if report_text:
        st.text_area("Extracted Report", value=report_text, height=300)

        # Generate and display the comparison as plain text
        comparison_text = generate_comparison_text(generated_report, ground_truth)
        st.text_area("Comparison", value=comparison_text, height=300)

        # Display identified errors
        st.write("Identified Errors:", errors)

        # BLEU Score
        bleu_score = calculate_bleu(generated_report, ground_truth)
        st.metric("BLEU SCORE", f"{bleu_score:.2f}")
import requests

def query_google_gemini(api_key, report, ground_truth):

 def query_google_gemini(api_key, generated_report, ground_truth):
    url = "https://google-gemini-api.example.com/v1/evaluate"
    payload = {
        "generated_report": report,
        "generated_report": generated_report,
        "ground_truth": ground_truth
    }
    headers = { "Autherization: f Bearer {api_key}"}
    headers = { "Authorization": f"Bearer {api_key}"}
    response = requests.post(url, json=payload, headers=headers)
    return response.json
    return response.json()

api_key = "AIzaSyDjqgQcMvCBqA8A9CrfQxUEn4rRu5ooPcM"
gemini_reposne = query_google_gemini(api_key,generated_report,ground_truth)
gemini_reposne = query_google_gemini(api_key, generated_report, ground_truth)
st.write("Gemini Evaluation Response:" ,gemini_reposne)