# comparators.py
import torch
from sentence_transformers import util

from normalizers import normalize_title

def title_match_scores(resume_titles, job_title, bi_model, logger=None):
    # Normalize titles first
    norm_job     = normalize_title(job_title)
    norm_resumes = [normalize_title(rt) for rt in resume_titles if rt]

    # Bi-encoder (MiniLM)
    if norm_resumes:
        emb_res = bi_model.encode(norm_resumes, convert_to_tensor=True)
        emb_job = bi_model.encode([norm_job],   convert_to_tensor=True)
        sims    = util.cos_sim(emb_res, emb_job).squeeze(1)  # tensor of shape (n,)
        best_idx = int(torch.argmax(sims))
        best_bi  = float(sims[best_idx])
        best_pair = (norm_resumes[best_idx], norm_job)

        # Convert to percent
        best_bi_pct = max(best_bi, 0.0) * 100.0

        # Log if you passed a streamlit logger or standard logger
        if logger:
            logger(f"MiniLM compared   → {best_pair[0]!r} vs {best_pair[1]!r} = {best_bi_pct:.2f}%")
    else:
        best_bi_pct = 0.0
        best_pair   = (None, norm_job)

    # Cross-encoder (RoBERTa-STS)
    
    #pairs = [[rt, norm_job] for rt in norm_resumes]
    #if pairs:
    #    ce_scores  = cross_encoder.predict(pairs)
    #   best_ce    = max(ce_scores)
    #    best_ce_pct= (best_ce / 5.0) * 100.0
    #else:
    #    best_ce_pct = 0.0

    #return best_bi_pct, best_ce_pct, best_pair
    
    return best_bi_pct, best_pair