import streamlit as st
import os
import pdfplumber
import re
import json
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from ollama import Client
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from pathlib import Path
from comparators import title_match_scores
from normalizers  import normalize_skills, normalize_soft_skills, normalize_title
import time


st.set_page_config(page_title="Resume vs Job Description", layout="centered")

models_root = Path("..") / "models"

@st.cache_resource
def load_model():
    local_subdir = models_root / "all-MiniLM-L6-v2"

    if local_subdir.is_dir():
        # 1) already downloaded
        model_path = str(local_subdir)
    else:
        # 2) pull from HF and get back the true download path
        models_root.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=str(models_root),
            repo_type="model"
        )


    #load from disk (or from the freshly-downloaded folder)
    return SentenceTransformer(model_path)

# @st.cache_resource
# def load_cross_encoder():
#     local_subdir = models_root / "cross-encoder/stsb-roberta-large"
#     if local_subdir.is_dir():
#         # 1) already downloaded
#         model_path = str(local_subdir)
#     else:
#         # 2) pull from HF and get back the true download path
#         models_root.mkdir(parents=True, exist_ok=True)
#         model_path = snapshot_download(
#             repo_id="cross-encoder/stsb-roberta-large",
#             cache_dir=str(models_root),
#             repo_type="model"
#         )
#     return CrossEncoder("cross-encoder/stsb-roberta-large")
# cross_encoder = load_cross_encoder()

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
- years_of_experience (must be the total number of years)
- skills
- past_job_titles
- education (only write "Bachelor's, Master's, PhD in ...")
- soft_skills

Example output:
{{
  "name": "Emma Liu",
  "years_of_experience": 3,
  "skills": ["Go", "Kubernetes", "PostgreSQL"],
  "past_job_titles": ["Platform Engineer", "Backend Developer"],
  "education": "Master's in Software Engineering",
  "soft_skills": ["teamplayer", "problem solver"]
}}

Resume:
{resume_text}

Respond only with valid JSON. Do not include any extra text or explanation.
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
- required_experience (must be a number of years)
- required_skills
- preferred_skills
- education (only write "Bachelor's, Master's, PhD in ...")
- soft_skills

Example output:
{{
  "title": "Backend Engineer",
  "required_experience": 2,
  "required_skills": ["Python", "AWS", "PostgreSQL"],
  "preferred_skills": ["Kubernetes", "Docker"],
  "education": "Bachelor's in Computer Science",
  "soft_skills": ["communication", "problem-solving"]
}}

Job Description:
{jd_text}

Respond only with valid JSON. Do not include any additional explanation or formatting.
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

    def sim_list(list1, list2, threshold=0.6):
        matches = []
        for item1 in list1:
            for item2 in list2:
                if sim(item1, item2) >= threshold:
                    matches.append((item1, item2))
        return matches

    #resume_skills = resume.get("skills", [])
    #required_skills = job.get("required_skills", [])
    #preferred_skills = job.get("preferred_skills", [])

    resume_skills    = normalize_skills(resume.get("skills", []))
    required_skills  = normalize_skills(job.get("required_skills", []))
    preferred_skills = normalize_skills(job.get("preferred_skills", []))

    def score_list_against_list(src, target, threshold=0.7):
        hits = 0
        for t in target:
            if any(sim(s, t) >= threshold for s in src):
                hits += 1
        return hits / max(1, len(target))

    # 3) Now score
    required_score  = score_list_against_list(resume_skills, required_skills)
    preferred_score = score_list_against_list(resume_skills, preferred_skills)



    required_matches = sim_list(resume_skills, required_skills)
    preferred_matches = sim_list(resume_skills, preferred_skills)

    required_score = len(required_matches) / max(1, len(required_skills))
    preferred_score = len(preferred_matches) / max(1, len(preferred_skills))
    skill_score = (required_score * 0.8) + (preferred_score * 0.2)

    #resume_exp = resume.get("years_of_experience", 0)
    #job_exp = job.get("required_experience", 0)
    #if isinstance(resume_exp, str):
    #    resume_exp = int(''.join(filter(str.isdigit, resume_exp)) or 0)

    #exp_score = 1.0 if resume_exp >= job_exp else 0.5 if resume_exp >= job_exp * 0.75 else 0.0



    # Pull out raw values
    raw_resume_exp = resume.get("years_of_experience", 0)
    raw_job_exp    = job.get("required_experience", 0)

    # Log them before any casting
    #st.write("🔍 DEBUG raw_resume_exp:", raw_resume_exp, "(", type(raw_resume_exp), ")")
    #st.write("🔍 DEBUG raw_job_exp:   ", raw_job_exp,    "(", type(raw_job_exp),    ")")

    # (Your existing parsing logic here)
    if isinstance(raw_resume_exp, str):
        resume_exp = int(''.join(filter(str.isdigit, raw_resume_exp)) or 0)
    else:
        resume_exp = raw_resume_exp

    if isinstance(raw_job_exp, str):
        job_exp = int(''.join(filter(str.isdigit, raw_job_exp)) or 0)
    else:
        job_exp = raw_job_exp

    # Log the parsed numbers
    #st.write("🔍 DEBUG parsed resume_exp:", resume_exp, "(", type(resume_exp), ")")
    #st.write("🔍 DEBUG parsed job_exp:   ", job_exp,    "(", type(job_exp),    ")")
    try:
        if resume_exp >= job_exp:
            exp_score = 1.0
        elif resume_exp >= job_exp * 0.75:
            exp_score = 0.5
        else:
            exp_score = 0.0
    except TypeError as e:
        st.error(f"TypeError during comparison: {e}")
        st.error(f"resume_exp={resume_exp!r} ({type(resume_exp)})  job_exp={job_exp!r} ({type(job_exp)})")
        raise

     
    # resume_titles = resume.get("past_job_titles", [])
    # job_title = job.get("title", "")
    
    # # Build all pairs of (resume_title, job_title)
    # pairs = [[rt, job_title] for rt in resume_titles if rt]
    # if pairs:
    #     # 2) Run the cross-encoder (returns a list of floats, typically 0–5)
    #     ce_scores = cross_encoder.predict(pairs)

    #     # 3) Take the maximum—as the “best match”
    #     best = max(ce_scores)

    #     # 4) Normalize into [0,1] (since STS models score ~0–5)
    #     # best_ce is the raw CE score (0–5)
    #     if best >= 3.5:
    #         title_score = 1.0
    #     elif best >= 2.0:
    #         title_score = 0.5
    #     else:
    #         title_score = 0.0
    

    bi_pct, title_pair = title_match_scores(
        resume_info.get("past_job_titles", []),
        job_info   .get("title",         ""),
        model,
        logger=lambda msg: st.write("🔍 DEBUG:", msg)
    )
    st.write(f"🔍 Best Title Pair (MiniLM): {title_pair[0]!r} ↔ {title_pair[1]!r} @ {bi_pct:.2f}%")

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


    raw_resume_soft = resume.get("soft_skills", [])
    raw_job_soft    = job.get("soft_skills", [])

    resume_soft_skills = normalize_soft_skills(raw_resume_soft, model, threshold=0.4)
    job_soft_skills    = normalize_soft_skills(raw_job_soft,    model, threshold=0.4)


    # now compare resume_soft_skills vs. job_soft_skills just like you do for required_skills
    soft_matches = sim_list(resume_soft_skills, job_soft_skills)
    soft_score = len(soft_matches) / max(1, len(job_soft_skills))

    bi_frac = bi_pct / 100.0

    final_score = (
        skill_score * 0.4 +
        exp_score * 0.2 +
        #title_score * 0.2 +
        bi_frac * 0.2 +
        edu_score * 0.1 +
        soft_score * 0.1
    )

    return {
        "required_skills":   required_score,
        "preferred_skills":  preferred_score,
        "overall_skills":    skill_score,
        "experience":        exp_score,
        # (you can keep the old title_score if you want)
        "title_bi":          bi_pct,
        #"title_ce":          ce_pct,
        "education":         edu_score,
        "soft_skills":       soft_score,
        "final":             final_score
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

            try:
                t0=time.time()
                print(f"[LOG]: Ollama Mistral call started")
                resume_info = extract_resume_info(resume_clean)
                job_info = extract_job_description_info(job_clean)
                st.write(f"✅ Ollama Mistral call took {time.time() - t0:.1f}s")
                print(f"[LOG]: Ollama Mistral call finished and took {time.time() - t0:.1f}s")

                #score = compare_resume_and_job(resume_info, job_info, model)
                st.success("✅ Comparison Complete")
                t1=time.time()
                print(f"[LOG]: MiniLM embeddings call started")
                scores = compare_resume_and_job(resume_info, job_info, model)
                st.write(f"✅ MiniLM embeddings took {time.time() - t1:.1f}s")
                print(f"[LOG]: MiniLM embeddings call finished and took {time.time() - t1:.1f}s")
                # Final
                st.success("✅ Comparison Complete")
                st.metric("🎯 Total Match", f"{scores['final']*100:.2f}%")

                # Breakdown: 2 columns for example
                col1, col2 = st.columns(2)
                col1.metric("🛠 Required Skills",f"{scores['required_skills']*100:.2f}%")
                col1.metric("⭐ Preferred Skills",f"{scores['preferred_skills']*100:.2f}%")
                col1.metric("🔧 Overall Skills",f"{scores['overall_skills']*100:.2f}%")
                col1.metric("🔡 Title Match (MiniLM)",f"{scores['title_bi']:.2f}%")

                #col2.metric("🔡 Title Match (Cross-Encoder)",f"{scores['title_ce']:.2f}%")
                col2.metric("📅 Experience",f"{scores['experience']*100:.2f}%")
                #col2.metric("🏷 Job Title Match",f"{scores['title']*100:.2f}%")
                col2.metric("🎓 Education",f"{scores['education']*100:.2f}%")
                col2.metric("🤝 Soft Skills",f"{scores['soft_skills']*100:.2f}%")

                
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
