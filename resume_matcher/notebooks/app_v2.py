import streamlit as st
import pdfplumber
import re
import json
from sentence_transformers import SentenceTransformer, util
from ollama import Client

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

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\n•\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Use Ollama to extract info from resume
def extract_resume_info(resume_text: str, model_name: str = 'mistral') -> dict:
    client = Client()
    prompt = f"""
Extract the following fields from the resume below and return them as a valid JSON object:

- name
- years_of_experience
- skills
- past_job_titles
- education
- soft_skills

Resume:
{resume_text}

Respond only with valid JSON.
"""
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response['message']['content'])

# Use Ollama to extract info from job description
def extract_job_description_info(jd_text: str, model_name: str = 'mistral') -> dict:
    client = Client()
    prompt = f"""
Extract the following fields from the job description and return them as a valid JSON object:

- title
- required_experience
- required_skills
- preferred_skills
- education
- soft_skills

Job Description:
{jd_text}

Respond only with valid JSON.
"""
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response['message']['content'])

# Compare resume and job JSON
def compare_resume_and_job(resume: dict, job: dict, model, explain: bool = False) -> float:
    def sim(a, b):
        return util.cos_sim(model.encode(a, convert_to_tensor=True), model.encode(b, convert_to_tensor=True)).item()

    def sim_list(list1, list2, threshold=0.7):
        matches = []
        for item1 in list1:
            for item2 in list2:
                if sim(item1, item2) >= threshold:
                    matches.append((item1, item2))
        return matches

    resume_skills = resume.get("skills", [])
    required_skills = job.get("required_skills", [])
    preferred_skills = job.get("preferred_skills", [])

    required_matches = sim_list(resume_skills, required_skills)
    preferred_matches = sim_list(resume_skills, preferred_skills)

    required_score = len(required_matches) / max(1, len(required_skills))
    preferred_score = len(preferred_matches) / max(1, len(preferred_skills))
    skill_score = (required_score * 0.8) + (preferred_score * 0.2)

    resume_exp = resume.get("years_of_experience", 0)
    job_exp = job.get("required_experience", 0)
    if isinstance(resume_exp, str):
        resume_exp = int(''.join(filter(str.isdigit, resume_exp)) or 0)

    exp_score = 1.0 if resume_exp >= job_exp else 0.5 if resume_exp >= job_exp * 0.75 else 0.0

    resume_titles = resume.get("past_job_titles", [])
    job_title = job.get("title", "")
    title_score = 0.0
    for rt in resume_titles:
        if sim(rt, job_title) > 0.75:
            title_score = 1.0
            break

    def simplify_education(text):
        text = text.lower()
        text = re.sub(r"bachelor(?:'s)?|bsc", "bachelor", text)
        text = re.sub(r"master(?:'s)?|msc", "master", text)
        text = re.sub(r"(in|of)", "", text)
        text = re.sub(r"[,\.]", "", text)
        keywords = ["bachelor", "master", "computer science", "mathematics", "data science", "engineering"]
        return ' '.join([kw for kw in keywords if kw in text])

    resume_edu_clean = simplify_education(resume.get("education", ""))
    job_edu_clean = simplify_education(job.get("education", ""))
    edu_score = sim(resume_edu_clean, job_edu_clean) if resume_edu_clean and job_edu_clean else 0.0
    edu_score = 1.0 if edu_score > 0.7 else 0.0

    final_score = (
        skill_score * 0.5 +
        exp_score * 0.2 +
        title_score * 0.2 +
        edu_score * 0.1
    )

    return round(final_score * 100, 2)

# Compare button functionality using session state
if "clicked" not in st.session_state:
    st.session_state["clicked"] = False

def compare():
    st.session_state["clicked"] = True

st.button("Compare", on_click=compare)

if st.session_state["clicked"]:
    if resume_file and job_description:
        with st.spinner("Extracting and comparing..."):
            resume_text_raw = extract_text_from_pdf(resume_file)
            resume_clean = clean_text(resume_text_raw)
            job_clean = clean_text(job_description)

            try:
                resume_info = extract_resume_info(resume_clean)
                job_info = extract_job_description_info(job_clean)
                score = compare_resume_and_job(resume_info, job_info, model)
                st.success("✅ Comparison Complete")
                st.metric(label="🎯 Match Score", value=f"{score:.2f}%")
                with st.expander("🔍 Extracted Resume Info"):
                    st.json(resume_info)
                with st.expander("📋 Extracted Job Info"):
                    st.json(job_info)
            except Exception as e:
                st.error(f"❌ Error during comparison: {e}")
                st.session_state["clicked"] = False
    else:
        st.warning("Please upload a resume and paste a job description.")
        st.session_state["clicked"] = False
