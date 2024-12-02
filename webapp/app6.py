import streamlit as st
import openai
import pdfplumber

# Set your OpenAI API Key
openai.api_key = "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"

# Streamlit page configuration
st.set_page_config(page_title="Radiology Report Correction", page_icon="🧬", layout="wide")
st.title("Radiology Report Correction with AI")
st.subheader("Automatically correct grammar and biological terms in radiology reports")

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

# Function to call OpenAI API and correct the report
def correct_radiology_report(report_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use gpt-4 if needed
            messages=[
                {"role": "system", "content": "You are an expert assistant that corrects grammar and ensures accurate biological terms in radiology reports."},
                {"role": "user", "content": f"Correct the grammar and biological terms in this radiology report:\n\n{report_text}"}
            ],
            max_tokens=1500,  # Adjust based on your needs
            temperature=0.2  # Low temperature for focused output
        )
        corrected_report = response['choices'][0]['message']['content']
        return corrected_report
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return None

# Streamlit UI for file upload
uploaded_file = st.file_uploader("Upload Radiology Report (PDF/TXT)", type=["pdf", "txt"])

if uploaded_file:
    # Extract text from the uploaded file
    report_text = extract_text_from_file(uploaded_file)
    
    if report_text:
        # Display the extracted report
        st.text_area("Extracted Report", value=report_text, height=300)

        # Correct the report using AI
        corrected_report = correct_radiology_report(report_text)

        if corrected_report:
            # Display the corrected report
            st.text_area("Corrected Report", value=corrected_report, height=300)
