'''
comparators.py

This module provides functionality to compute similarity scores between job titles
and resume titles using a bi-encoder model. It normalizes titles, encodes them,
and returns the best matching pair along with a similarity percentage.
'''
import torch
from sentence_transformers import util

from normalizers import normalize_title

def title_match_scores(resume_titles, job_title, bi_model, logger=None):
    '''
    Compute similarity scores between a list of resume titles and a single job title.

    Parameters:
    resume_titles (list of str): List of titles extracted from resumes to compare.
    job_title (str): The job title to match against each resume title.
    bi_model: A bi-encoder model instance (e.g., SentenceTransformer) used to compute embeddings.
    logger (optional): Logger instance for debugging or informational messages (unused in this implementation).

    Returns:
    tuple:
        best_bi_pct (float): The highest cosine similarity score (in percentage) between any resume title and the job title.
        best_pair (tuple): A tuple containing the normalized resume title with the highest match and the normalized job title.
                           If no resume titles are provided, returns (None, normalized job title) with a similarity of 0.0.
    '''
    norm_job     = normalize_title(job_title)
    norm_resumes = [normalize_title(rt) for rt in resume_titles if rt]

    if norm_resumes:
        emb_res = bi_model.encode(norm_resumes, convert_to_tensor=True)
        emb_job = bi_model.encode([norm_job],   convert_to_tensor=True)
        sims    = util.cos_sim(emb_res, emb_job).squeeze(1)
        best_idx = int(torch.argmax(sims))
        best_bi  = float(sims[best_idx])
        best_pair = (norm_resumes[best_idx], norm_job)

        best_bi_pct = max(best_bi, 0.0) * 100.0

    else:
        best_bi_pct = 0.0
        best_pair   = (None, norm_job)

    
    return best_bi_pct, best_pair