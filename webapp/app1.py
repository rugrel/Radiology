import streamlit as st
import pdfplumber
import os
import requests
from difflib import unified_diff
from dotenv import load_dotenv
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu

# Load environment variables
load_dotenv()

# API Key for Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure the key is set in .env
GEMINI_API_URL = "https://gemini-api-url/v1/evaluate"  # Replace with actual URL

# Streamlit page configuration
st.set_page_config(page_title="Radiology Report Analysis", page_icon="🧬", layout="wide")
st.title("Visualize Radiology Report and Correction 🌡")
st.subheader("An App to help with Radiology Analysis using AI")

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

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

# Function to call Google Gemini API
def query_google_gemini(api_key, report_text, ground_truth_text):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "generated_report": report_text,
        "ground_truth": ground_truth_text,
    }
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying Google Gemini API: {e}")
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

# Improved error detection
def find_errors(generated, ground_truth):
    gen_doc = nlp(generated.lower())
    truth_doc = nlp(ground_truth.lower())

    missed_entities = [(ent.text, "Missed") for ent in truth_doc.ents if ent.text.lower() not in generated.lower()]
    extra_entities = [(ent.text, "Extra") for ent in gen_doc.ents if ent.text.lower() not in ground_truth.lower()]

    return missed_entities + extra_entities

# METRICS CALCULATION
def calculate_bleu(generated, ground_truth):
    return sentence_bleu([ground_truth.split()], generated.split())

# Streamlit UI for uploading file and viewing outputs
uploaded_file = st.file_uploader("Upload a Radiology Report (PDF or TXT)", type=["pdf", "txt"])
ground_truth = st.text_area("Enter the Ground Truth Report", "Patient has a normal chest X-ray. No fractures or lesions.")

if uploaded_file and ground_truth:
    # Extract report text
    report_text = extract_text_from_file(uploaded_file)
    if report_text:
        st.text_area("Extracted Report", value=report_text, height=300)

        # Call Google Gemini API
        gemini_response = query_google_gemini(GEMINI_API_KEY, report_text, ground_truth)
        if gemini_response:
            st.write("Google Gemini Response:")
            st.json(gemini_response)

        # Generate and display the comparison as plain text
        comparison_text = generate_comparison_text(report_text, ground_truth)
        st.text_area("Comparison", value=comparison_text, height=300)

        # Identify errors
        errors = find_errors(report_text, ground_truth)
        st.write("Identified Errors:", errors)

        # BLEU Score
        bleu_score = calculate_bleu(report_text, ground_truth)
        st.metric("BLEU SCORE", f"{bleu_score:.2f}")
