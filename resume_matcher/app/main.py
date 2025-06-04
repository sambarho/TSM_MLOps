import streamlit as st
import os
import pandas as pd
import re
import json
from sentence_transformers import util
from comparators import title_match_scores
from normalizers  import normalize_skills, normalize_soft_skills
from model_loader import load_model
from extractors import extract_text_from_pdf, clean_text
from ollama_api import extract_resume_info, extract_job_description_info
import time
from database import MatchRecord, Session, save_match_record  

st.set_page_config(page_title="Resume vs Job Description", layout="centered")

model = load_model()
st.title("📄 Resume & Job Description Matcher")

resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

job_description = st.text_area("Paste the job description below", height=300)

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

    resume_skills    = normalize_skills(resume.get("skills", []))
    required_skills  = normalize_skills(job.get("required_skills", []))
    preferred_skills = normalize_skills(job.get("preferred_skills", []))

    def score_list_against_list(src, target, threshold=0.7):
        hits = 0
        for t in target:
            if any(sim(s, t) >= threshold for s in src):
                hits += 1
        return hits / max(1, len(target))

    required_score  = score_list_against_list(resume_skills, required_skills)
    preferred_score = score_list_against_list(resume_skills, preferred_skills)

    # for scoring we only care "did I match each JD skill at least once?"
    def binary_score(src, target, threshold=0.7):
            hits = 0
            for t in target:
                # if ANY resume skill is close enough to this JD skill, count it
                if any(util.cos_sim(model.encode(s), model.encode(t)) >= threshold for s in src):
                    hits += 1
            return hits / max(1, len(target))

    required_score  = binary_score(resume_skills, required_skills)
    preferred_score = binary_score(resume_skills, preferred_skills)
    # overall skills = weighted average, still bounded [0..1]
    skill_score     = (required_score * 0.8) + (preferred_score * 0.2)

    # raw values
    raw_resume_exp = resume.get("years_of_experience", 0)
    raw_job_exp    = job.get("required_experience", 0)

    # log them before any casting
    #st.write("DEBUG raw_resume_exp:", raw_resume_exp, "(", type(raw_resume_exp), ")")
    #st.write("DEBUG raw_job_exp:   ", raw_job_exp,    "(", type(raw_job_exp),    ")")

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


    bi_pct, title_pair = title_match_scores(
        resume_info.get("past_job_titles", []),
        job_info   .get("title",         ""),
        model,

    )
    ### st.write(f"🔍 Best Title Pair (MiniLM): {title_pair[0]!r} ↔ {title_pair[1]!r} @ {bi_pct:.2f}%") for demo

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

    soft_matches = sim_list(resume_soft_skills, job_soft_skills)
    soft_score = len(soft_matches) / max(1, len(job_soft_skills))

    bi_frac = bi_pct / 100.0

    final_score = (
        skill_score * 0.4 +
        exp_score * 0.2 +
        bi_frac * 0.2 +
        edu_score * 0.1 +
        soft_score * 0.1
    )

    return {
        "required_skills":   required_score,
        "preferred_skills":  preferred_score,
        "overall_skills":    skill_score,
        "experience":        exp_score,
        "title_bi":          bi_pct,
        "education":         edu_score,
        "soft_skills":       soft_score,
        "final":             final_score
    }


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
                # demo
                ## st.write(f"✅ Ollama Mistral call took {time.time() - t0:.1f}s") 
                ##print(f"[LOG]: Ollama Mistral call finished and took {time.time() - t0:.1f}s")

                ##st.success("✅ Comparison Complete")
                t1=time.time()
                print(f"[LOG]: MiniLM embeddings call started")
                scores = compare_resume_and_job(resume_info, job_info, model)
                ##st.write(f"✅ MiniLM embeddings took {time.time() - t1:.1f}s")
                print(f"[LOG]: MiniLM embeddings call finished and took {time.time() - t1:.1f}s")

                st.success("✅ Comparison Complete")
                st.metric("🎯 Total Match", f"{scores['final']*100:.2f}%")

                col1, col2 = st.columns(2)
                col1.metric("🛠 Required Skills",f"{scores['required_skills']*100:.2f}%")
                col1.metric("⭐ Preferred Skills",f"{scores['preferred_skills']*100:.2f}%")
                col1.metric("🔧 Overall Skills",f"{scores['overall_skills']*100:.2f}%")
                ### col1.metric("🔡 Title Match (MiniLM)",f"{scores['title_bi']:.2f}%") demo

                col2.metric("📅 Experience",f"{scores['experience']*100:.2f}%")
                col2.metric("🎓 Education",f"{scores['education']*100:.2f}%")
                col2.metric("🤝 Soft Skills",f"{scores['soft_skills']*100:.2f}%")

                ### DEMO ###
                #with st.expander("🔍 Extracted Resume Info"):
                #    st.json(resume_info)
                #with st.expander("📋 Extracted Job Info"):
                #    st.json(job_info)
                resume_df = pd.DataFrame({
                    "Field": [
                        "Name",
                        "Years of Experience",
                        "Education",
                        "Skills",
                        "Soft Skills",
                        "Past Job Titles",
                    ],
                    "Value": [
                        resume_info.get("name", ""),
                        resume_info.get("years_of_experience", ""),
                        resume_info.get("education", ""),
                        ", ".join(resume_info.get("skills", [])),
                        ", ".join(resume_info.get("soft_skills", [])),
                        ", ".join(resume_info.get("past_job_titles", [])),
                    ]
                })
                st.subheader("🔍 Extracted Resume Info")
                st.table(resume_df)

                # job Info table
                job_df = pd.DataFrame({
                    "Field": [
                        "Title",
                        "Required Experience",
                        "Education",
                        "Required Skills",
                        "Preferred Skills",
                        "Soft Skills",
                    ],
                    "Value": [
                        job_info.get("title", ""),
                        job_info.get("required_experience", ""),
                        job_info.get("education", ""),
                        ", ".join(job_info.get("required_skills", [])),
                        ", ".join(job_info.get("preferred_skills", [])),
                        ", ".join(job_info.get("soft_skills", [])),
                    ]
                })
                st.subheader("📋 Extracted Job Description Info")
                st.table(job_df)
                
                thresh = 0.7
                resume_sk = normalize_skills(resume_info.get("skills", []))
                req_sk    = normalize_skills(job_info.get("required_skills", []))
                pref_sk   = normalize_skills(job_info.get("preferred_skills", []))

                def find_missing(target_skills, source_skills):
                    missing = []
                    for t in target_skills:
                        # count as “covered” if any resume skill is semantically similar
                        if not any(util.cos_sim(model.encode(s), model.encode(t)).item() >= thresh
                                for s in source_skills):
                            missing.append(t)
                    return missing

                missing_required  = find_missing(req_sk, resume_sk)
                missing_preferred = find_missing(pref_sk, resume_sk)

                if missing_required:
                    st.error(
                        "🚨 You’re missing these **required** skills:\n\n" +
                        ", ".join(missing_required)
                    )
                if missing_preferred:
                    st.info(
                        "💡 Consider adding or learning these **preferred** skills:\n\n" +
                        ", ".join(missing_preferred)
                    )


            except Exception as e:
                st.error(f"❌ Error during comparison: {e}")
                st.session_state["clicked"] = False
        save_match_record(resume_file, job_description, resume_info, job_info, scores)  #<-- add

        
    else:
        st.warning("Please upload a resume and paste a job description.")
        st.session_state["clicked"] = False

################################################
# View previous comparisons in a neat format
st.subheader("📂 View Previous Comparisons")

session = Session()
records = session.query(MatchRecord).order_by(MatchRecord.timestamp.desc()).all()

for rec in records:
    job_info = json.loads(rec.job_info)
    scores = json.loads(rec.comparison_scores)
    
    job_title = job_info.get('title', 'No title available')
    overall_score = scores.get('final', 0) * 100

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