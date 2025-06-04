import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util
import os

st.set_page_config(page_title="Resume vs Job Description", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("../models/all-MiniLM-L6-v2")

model = load_model()


st.title("📄 Resume & Job Description Matcher")

# Upload PDF resume
resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Paste the job description below", height=300)

# Custom PDF extraction
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

# Basic text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\n•\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compare_button():
    st.button("Compare", key="compare_button_1", on_click=compare)

if "clicked" not in st.session_state:
    st.session_state["clicked"] = False

def compare():
    st.session_state["clicked"] = True

if st.session_state["clicked"]:
    if resume_file and job_description:
        resume_text_raw = extract_text_from_pdf(resume_file)
        resume_clean = clean_text(resume_text_raw)
        job_clean = clean_text(job_description)
    
        # Embed cleaned text
        resume_embed = model.encode(resume_clean, convert_to_tensor=True)
        job_embed = model.encode(job_clean, convert_to_tensor=True)
    
        similarity_score = util.cos_sim(resume_embed, job_embed).item()
    
        st.subheader("🔍 Similarity Score")
        st.metric(label="Match Percentage", value=f"{similarity_score * 100:.2f}%")
    
        with st.expander("🔎 Show extracted resume text"):
            st.write(resume_clean)
        with st.expander("🔎 Show extracted Job text"):
            st.write(job_clean)
    else:
        st.warning("Please upload a resume and paste a job description.")
        st.session_state["clicked"] = False

compare_button()
