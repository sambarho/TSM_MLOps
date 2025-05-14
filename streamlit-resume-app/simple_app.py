import streamlit as st
import os
import pdfplumber
import re
import json
from database import MatchRecord, Session, save_match_record    #<-- add

st.set_page_config(page_title="Resume vs Job Description", layout="centered")
st.title("📄 Resume & Job Description Matcher (Test Mode)")

# Upload PDF resume
resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Paste the job description below:", height=300)

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

# Dummy resume info extractor
def extract_resume_info(resume_text):
    return {
        "name": "Test User",
        "years_of_experience": 4,
        "skills": ["Python", "SQL", "AWS"],
        "past_job_titles": ["Software Engineer", "Data Analyst"],
        "education": "Bachelor's in Computer Science",
        "soft_skills": ["communication", "teamwork"]
    }

# Dummy job description info extractor
def extract_job_description_info(jd_text):
    return {
        "title": "Backend Developer",
        "required_experience": 3,
        "required_skills": ["Python", "Docker"],
        "preferred_skills": ["Kubernetes", "PostgreSQL"],
        "education": "Bachelor's in Computer Science",
        "soft_skills": ["communication", "problem-solving"]
    }

# Dummy comparison
def compare_resume_and_job(resume, job):
    return {
        "required_skills": 0.8,
        "preferred_skills": 0.6,
        "overall_skills": 0.75,
        "experience": 1.0,
        "title_bi": 85.0,
        "education": 1.0,
        "soft_skills": 0.9,
        "final": 0.85
    }

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

            resume_info = extract_resume_info(resume_clean)
            job_info = extract_job_description_info(job_clean)
            scores = compare_resume_and_job(resume_info, job_info)

            st.success("✅ Comparison Complete")
            st.metric("🎯 Total Match", f"{scores['final']*100:.2f}%")

            col1, col2 = st.columns(2)
            col1.metric("🛠 Required Skills", f"{scores['required_skills']*100:.2f}%")
            col1.metric("⭐ Preferred Skills", f"{scores['preferred_skills']*100:.2f}%")
            col1.metric("🔧 Overall Skills", f"{scores['overall_skills']*100:.2f}%")
            col1.metric("🔡 Title Match (Mock)", f"{scores['title_bi']:.2f}%")

            col2.metric("📅 Experience", f"{scores['experience']*100:.2f}%")
            col2.metric("🎓 Education", f"{scores['education']*100:.2f}%")
            col2.metric("🤝 Soft Skills", f"{scores['soft_skills']*100:.2f}%")

            with st.expander("🔍 Extracted Resume Info"):
                st.json(resume_info)
            with st.expander("📋 Extracted Job Info"):
                st.json(job_info)

            # Save to database
            save_match_record(resume_file, job_description, resume_info, job_info, scores)  #<-- add
            
    else:
        st.warning("Please upload a resume and paste a job description.")
        st.session_state["clicked"] = False

################## add ########################
# View previous comparisons in a neat format
st.subheader("📂 View Previous Comparisons")

session = Session()
records = session.query(MatchRecord).order_by(MatchRecord.timestamp.desc()).all()

for rec in records:
    job_info = json.loads(rec.job_info)
    scores = json.loads(rec.comparison_scores)
    
    job_title = job_info.get('title', 'No title available')
    overall_score = scores.get('final', 0) * 100

    # This is the only expander now
    with st.expander(f"📝 {job_title} — {overall_score:.2f}% match"):
        st.write(f"📄 **Resume File Name**: {rec.resume_name}")

        st.write("📊 **Comparison Scores**:")

        col1, col2, col3 = st.columns(3)

        col1.metric("🛠 Required Skills", f"{scores['required_skills']*100:.2f}%")
        col1.metric("⭐ Preferred Skills", f"{scores['preferred_skills']*100:.2f}%")
        col1.metric("🔧 Overall Skills", f"{scores['overall_skills']*100:.2f}%")

        col2.metric("📅 Experience", f"{scores['experience']*100:.2f}%")
        col2.metric("🎓 Education", f"{scores['education']*100:.2f}%")
        col2.metric("🤝 Soft Skills", f"{scores['soft_skills']*100:.2f}%")

        col3.metric("🔡 Title Match", f"{scores['title_bi']:.2f}%")
        col3.metric("🎯 Final Score", f"{scores['final']*100:.2f}%")

        st.write(f"🕓 **Date and Time**: {rec.timestamp.strftime('%Y-%m-%d %H:%M')}")

session.close()
################################################
