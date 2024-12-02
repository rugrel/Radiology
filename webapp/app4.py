import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pdfplumber
import os
from difflib import unified_diff
from dotenv import load_dotenv
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu

# Load environment variables (if you have any for your system)
load_dotenv()

# Load BioClinical BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Streamlit page configuration
st.set_page_config(page_title="Radiology Report Analysis", page_icon="🧬", layout="wide")
st.title("Visualize Radiology Report and Correction 🌡")
st.subheader("An App to help with Radiology Analysis using AI")

# Function to extract text from uploaded file (PDF/TXT)
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

# Function to tokenize a report
def tokenize_report(report_text):
    inputs = tokenizer(report_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    return inputs

# Function to get entities from a report (from BioClinical BERT)
def get_entities_from_report(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    
    # Get predicted entity labels
    predicted_entities = predictions[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())
    
    # Map tokens to labels
    labels = [model.config.id2label[label] for label in predicted_entities]
    
    return tokens, labels

# Function to find errors in the report (comparison with ground truth)
def find_errors_in_report(report_text, ground_truth):
    inputs_report = tokenize_report(report_text)
    inputs_truth = tokenize_report(ground_truth)
    
    # Get entities from report and ground truth
    tokens_report, entities_report = get_entities_from_report(inputs_report)
    tokens_truth, entities_truth = get_entities_from_report(inputs_truth)
    
    # Identify missed and extra entities
    missed_entities = set(entities_truth) - set(entities_report)
    extra_entities = set(entities_report) - set(entities_truth)
    
    # Return a list of missed and extra entities
    missed = [tokens_truth[i] for i in range(len(tokens_truth)) if entities_truth[i] in missed_entities]
    extra = [tokens_report[i] for i in range(len(tokens_report)) if entities_report[i] in extra_entities]
    
    return missed, extra

# Function to generate a plain text comparison between generated and ground truth report
def generate_comparison_text(generated_report, ground_truth):
    diff = unified_diff(
        ground_truth.splitlines(), 
        generated_report.splitlines(), 
        fromfile="Ground Truth", 
        tofile="Generated Report", 
        lineterm=""
    )
    return "\n".join(diff)

# BLEU score function
def calculate_bleu(generated, ground_truth):
    return sentence_bleu([ground_truth.split()], generated.split())

# Function to apply corrections (in a simple manner: adding missed entities)
def apply_corrections(report_text, missed_entities, extra_entities):
    corrected_report = report_text
    
    # Simple correction logic: add missed entities into the report
    for entity in missed_entities:
        corrected_report += f" [Missing Entity: {entity}]"
    
    # You can also modify the extra entities (simple placeholder approach)
    for entity in extra_entities:
        corrected_report = corrected_report.replace(entity, f"[Extra Entity: {entity}]")
    
    return corrected_report

# Streamlit UI for file upload and ground truth input
uploaded_file = st.file_uploader("Upload Radiology Report (PDF/TXT)", type=["pdf", "txt"])
ground_truth = st.text_area("Enter Ground Truth Report (for comparison)", height=200)

if uploaded_file:
    # Extract text from the uploaded file
    report_text = extract_text_from_file(uploaded_file)
    
    if report_text:
        st.text_area("Extracted Report", value=report_text, height=300)

        # Check if ground truth is provided
        if ground_truth:
            # Compare the uploaded report with the provided ground truth
            comparison_text = generate_comparison_text(report_text, ground_truth)
            st.text_area("Comparison", value=comparison_text, height=300)

            # Find errors (missed and extra entities)
            missed_entities, extra_entities = find_errors_in_report(report_text, ground_truth)
            st.write("Missed Entities:", missed_entities)
            st.write("Extra Entities:", extra_entities)

            # Apply corrections and display the corrected report
            corrected_report = apply_corrections(report_text, missed_entities, extra_entities)
            st.text_area("Corrected Report", value=corrected_report, height=300)

            # Calculate BLEU score for the report comparison
            bleu_score = calculate_bleu(report_text, ground_truth)
            st.metric("BLEU SCORE", f"{bleu_score:.2f}")
        else:
            st.warning("Please provide a ground truth report for comparison.")
