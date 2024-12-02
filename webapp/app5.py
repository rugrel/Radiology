import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pdfplumber
import os
from difflib import unified_diff
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu

# Load environment variables (if you have any for your system)
load_dotenv()

# Load BioClinical BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Streamlit page configuration
st.set_page_config(page_title="Radiology Report Analysis", page_icon="🧬", layout="wide")
st.title("Radiology Report Error Highlighting 🌡")
st.subheader("AI-Powered Radiology Report Analysis")

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

# Function to find errors in the report
def find_errors_in_report(report_text, ground_truth):
    inputs_report = tokenize_report(report_text)
    inputs_truth = tokenize_report(ground_truth)
    
    # Get entities from report and ground truth
    tokens_report, entities_report = get_entities_from_report(inputs_report)
    tokens_truth, entities_truth = get_entities_from_report(inputs_truth)
    
    # Identify missed and extra entities
    missed_entities = set(entities_truth) - set(entities_report)
    extra_entities = set(entities_report) - set(entities_truth)
    
    # Return tokens corresponding to missed and extra entities
    missed = [tokens_truth[i] for i in range(len(tokens_truth)) if entities_truth[i] in missed_entities]
    extra = [tokens_report[i] for i in range(len(tokens_report)) if entities_report[i] in extra_entities]
    
    return missed, extra

# Function to highlight errors in the report
def highlight_errors(report_text, missed_entities, extra_entities):
    words = report_text.split()
    highlighted_text = []
    for word in words:
        if word in missed_entities:
            highlighted_text.append(f'<span style="background-color: yellow;">{word}</span>')
        elif word in extra_entities:
            highlighted_text.append(f'<span style="background-color: red;">{word}</span>')
        else:
            highlighted_text.append(word)
    return " ".join(highlighted_text)

# BLEU score function
def calculate_bleu(generated, ground_truth):
    return sentence_bleu([ground_truth.split()], generated.split())

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
            # Find errors (missed and extra entities)
            missed_entities, extra_entities = find_errors_in_report(report_text, ground_truth)

            # Highlight errors in the report
            highlighted_report = highlight_errors(report_text, missed_entities, extra_entities)

            # Display highlighted errors in the report
            st.markdown("### Highlighted Errors")
            st.markdown(f"<div style='font-family:monospace;'>{highlighted_report}</div>", unsafe_allow_html=True)

            # Calculate BLEU score for the report comparison
            bleu_score = calculate_bleu(report_text, ground_truth)
            st.metric("BLEU SCORE", f"{bleu_score:.2f}")
        else:
            st.warning("Please provide a ground truth report for comparison.")
