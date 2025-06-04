'''
ollama_api.py

This module provides utility functions to extract structured information from
resume text and job description text using the Ollama chat API. Each function
constructs a prompt for the specified model and returns parsed JSON output.
'''
from ollama import Client
import json

def extract_resume_info(resume_text: str, model_name: str = 'mistral') -> dict:
    '''
    Extract key fields from a resume using the Ollama chat model.
    '''
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

# use Ollama to extract info from job description
def extract_job_description_info(jd_text: str, model_name: str = 'mistral') -> dict:
    '''
    Extract key fields from a jb using the Ollama chat model.
    '''    
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

