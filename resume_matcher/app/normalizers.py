'''
normalizers.py

This module provides functions and data structures to normalize and canonicalize
skill and title strings. It includes:
- SKILL_ONTOLOGY: A mapping of canonical technical skills to their variant names.
- SOFT_SKILL_TAXONOMY: A mapping of canonical soft skills to their variant names.
- TITLE_SYNONYMS: Common job title synonyms for normalization.
- Functions to explode parenthesized skill lists, normalize technical skills,
  normalize soft skills using embedding similarity, and normalize job titles.
'''
import re
from sentence_transformers import util
import torch
# ------------------------------------------------------------------------------
# Technical Skill Ontology
#
# Maps a canonical skill key to a list of known variants/aliases.
# Used to map raw skill strings to a consistent canonical form.
# ------------------------------------------------------------------------------
SKILL_ONTOLOGY = {
    # Programming & Tech Stacks (existing + extended)
    "python":       ["python", "py", "jupyter", "jupyter notebook", "jupyterlab", "pandas", "numpy", "scipy", "matplotlib", "seaborn"],
    "javascript":   ["javascript", "js", "nodejs", "node", "typescript", "es6"],
    "html":         ["html", "css", "html5", "css3", "sass", "scss", "bootstrap", "tailwind"],
    "csharp":       ["csharp", "c#", ".net", ".net core", "asp.net", "blazor"],
    "c++":          ["c++", "c plus plus", "cpp"],
    "c":            ["c", "c programming", "ansi c"],
    "go":           ["go", "golang"],
    "ruby":         ["ruby", "rails", "ruby on rails"],
    "java":         ["java", "java 8", "java 11", "java 17", "spring", "spring boot", "maven", "gradle"],
    "flutter":      ["flutter", "dart"],
    "android":      ["android", "android sdk", "android studio", "kotlin"],
    "ios":          ["ios", "swift", "objective-c", "xcode"],
    "react":        ["react", "reactjs", "react.js", "react native", "next.js"],
    "angular":      ["angular", "angularjs", "angular 2", "angular 4", "angular 5", "angular 12"],
    "vue":          ["vue", "vuejs", "vue.js", "nuxt"],

    # Databases
    "sql":          ["sql", "sql server", "mysql", "mssql", "sqlite", "oracle", "db", "database", "sqlalchemy", "relational database"],
    "nosql":        ["nosql", "mongodb", "cassandra", "couchdb", "dynamodb", "redis", "documentdb"],
    "postgresql":   ["postgresql", "postgres", "psql"],

    # Data & ML
    "ml":           ["machine learning", "ml", "scikit-learn", "xgboost", "lightgbm", "catboost"],
    "dl":           ["deep learning", "dl", "tensorflow", "keras", "pytorch", "cnn", "rnn", "lstm", "transformer"],
    "nlp":          ["nlp", "natural language processing", "spacy", "nltk", "huggingface", "bert", "gpt"],
    "r":            ["r", "rstats", "tidyverse", "ggplot2", "shiny"],
    "matlab":       ["matlab", "simulink"],
    "sas":          ["sas", "statistical analysis system"],
    "excel":        ["excel", "vba", "spreadsheet", "microsoft excel"],
    "data_viz":     ["data visualization", "tableau", "powerbi", "lookml", "dash", "plotly"],

    # DevOps & Infrastructure
    "docker":       ["docker", "docker container", "docker-compose"],
    "kubernetes":   ["kubernetes", "k8s", "kube", "kubectl"],
    "devops":       ["devops", "ci/cd", "jenkins", "travis", "github actions", "gitlab ci", "circleci", "build pipeline"],
    "linux":        ["linux", "bash", "shell scripting", "unix", "zsh"],

    # Cloud Platforms
    "aws":          ["aws", "amazon web services", "ec2", "s3", "lambda", "cloudformation", "rds"],
    "azure":        ["azure", "microsoft azure", "azure devops", "azure functions"],
    "gcp":          ["gcp", "google cloud", "google cloud platform", "bigquery", "firebase"],
    "cloud":        ["cloud", "cloud computing", "cloud services", "cloud native", "multi-cloud", "hybrid cloud"],

    # Big Data & ETL
    "bigdata":      ["big data", "hadoop", "spark", "hive", "flink"],
    "data_engineering": ["data engineering", "etl", "airflow", "luigi", "dbt", "data pipeline"],

    # APIs & Integration
    "api":          ["api", "rest", "restful", "graphql", "openapi", "postman", "swagger"],
    "graphql":      ["graphql", "apollo client", "apollo server"],

    # Testing
    "testing":      ["unit testing", "integration testing", "pytest", "junit", "selenium", "cypress"],

    # Project Management & Collaboration
    "agile":        ["agile", "scrum", "kanban", "jira", "confluence"],
    "pm":           ["project management", "product owner", "pmo", "prince2", "pmp", "stakeholder management"],

    # Git & Versioning
    "git":          ["git", "github", "gitlab", "bitbucket", "version control"],

    # Architecture & Engineering
    "solution_architecture": ["solution architecture", "solution architect", "technical architect", "application architecture"],
    "enterprise_architecture": ["enterprise architecture", "TOGAF", "business capability modeling"],
    "system_engineering":     ["system engineering", "systems engineering", "system integration", "system lifecycle"],
    "cloud_engineering":      ["cloud engineering", "cloud engineer", "infrastructure as code", "terraform", "pulumi", "bicep"],
    "platform_engineering":   ["platform engineering", "internal developer platform", "platform team"],
    "site_reliability":       ["site reliability", "sre", "site reliability engineering", "incident response", "on-call", "service level objectives"],

    # Security
    "security_engineering": ["security engineering", "security engineer", "secure coding", "zero trust", "identity and access management", "iam", "sso", "oauth2", "jwt", "owasp", "penetration testing", "vulnerability scanning", "siem", "sast", "dast"],
    "cybersecurity":        ["cybersecurity", "information security", "infosec", "risk management", "iso 27001", "nist", "gdpr"],

    # Emerging Technologies
    "web3":         ["web3", "blockchain", "ethereum", "solidity", "smart contract"],
    "ai_engineering": ["ai engineering", "mlops", "llmops", "model deployment", "model monitoring", "mlflow", "onnx", "huggingface", "tuning foundation models"]
}



VARIANT_TO_CANONICAL = {
    variant: canon
    for canon, variants in SKILL_ONTOLOGY.items()
    for variant in variants
}

def explode_parens(skill_str):
    '''
    Split a skill string containing parentheses or comma/slash-separated lists into individual items.

    Parameters:
        skill_str (str): A raw skill string, possibly containing parentheses or separators.

    Returns:
        list[str]: A list of substrings, with parentheses and comma/slash splits exploded into separate items.
    '''
    base, *parens = re.split(r"\(|\)", skill_str)
    items = [base] + parens
    # split comma/slash lists too
    flat = []
    for item in items:
        flat += re.split(r"[,/]", item)
    return [i.strip() for i in flat if i.strip()]

def normalize_skills(raw_skills):
    '''
    Normalize a list of raw technical skill strings into canonical skill keys.

    Steps:
    1) For each raw skill string, call explode_parens() to handle parenthetical or combined lists.
    2) Lowercase and strip non-alphanumeric/+-/. characters.
    3) Map each cleaned string to its canonical key via VARIANT_TO_CANONICAL lookup.
    4) Deduplicate while preserving order.

    Parameters:
        raw_skills (list[str]): A list of skill strings extracted from a resume or job description.

    Returns:
        list[str]: A deduplicated list of canonical skill keys.
    '''
    normalized = []
    for s in raw_skills:
        # first, explode any parentheses into sub-skills
        parts = explode_parens(s)
        for part in parts:
            key = part.strip().lower()
            # allow plus, hash, and dot
            key = re.sub(r"[^a-z0-9\+#\. ]+", "", key)
            canon = VARIANT_TO_CANONICAL.get(key, key)
            if canon not in normalized:
                normalized.append(canon)
    return normalized

# ------------------------------------------------------------------------------
# Soft Skill Taxonomy
#
# Maps a canonical soft-skill key to its variant names. Used for exact lookup
# and embedding-based similarity if no exact match is found.
# ------------------------------------------------------------------------------
SOFT_SKILL_TAXONOMY = {
    "teamwork":            ["team player", "teamplayer","teamwork", "collaborates well", "collaboration", "works well with others"],
    "communication":       ["communication", "communicator", "verbal communication", "written communication", "presents ideas clearly", "active listening"],
    "problem-solving":     ["problem solving", "problem-solver", "analytical", "critical thinking", "troubleshooting", "logic-driven"],
    "adaptability":        ["adaptable", "flexible", "quick learner", "open to change", "resilient", "handles ambiguity"],
    "leadership":          ["leadership", "leader", "manages team", "mentoring", "coaching", "inspires others", "guides team"],
    "time-management":     ["time management", "organized", "prioritization", "meets deadlines", "efficient", "plans ahead"],
    "creativity":          ["creative", "innovation", "innovative", "out-of-the-box thinking", "original thinking"],
    "attention-to-detail": ["detail-oriented", "attention to detail", "accuracy", "precise", "quality-focused"],
    "empathy":             ["empathetic", "empathetic listener", "understands others", "emotional intelligence", "compassionate"],
    "initiative":          ["takes initiative", "self-starter", "proactive", "drives change", "motivated"],
    "accountability":      ["accountable", "ownership", "responsibility", "reliable", "dependable"],
    "conflict-resolution": ["conflict resolution", "mediates well", "resolves disagreements", "diplomatic", "negotiation"],
    "decision-making":     ["decision making", "decisive", "evaluates options", "judgment", "makes informed decisions"],
    "interpersonal":       ["interpersonal skills", "builds rapport", "relational", "people skills"],
    "customer-orientation": ["customer-oriented", "customer-centric", "client focus", "understands client needs"],
    "growth-mindset":      ["growth mindset", "eager to learn", "continuously improves", "embraces feedback"],
    "stress-management":   ["handles pressure", "stress tolerance", "stays calm under pressure", "composed"],
    "presentation":        ["presentation skills", "public speaking", "presents confidently", "engaging speaker"],
    "cross-cultural":      ["cultural awareness", "works in diverse teams", "inclusive", "open-minded", "global mindset"],
    "negotiation":         ["negotiation", "persuasive", "influences others", "finds compromise"]
}


# build reverse lookup: variant -> canonical
VARIANT_TO_CANONICAL_SOFT = {
    variant: canon
    for canon, variants in SOFT_SKILL_TAXONOMY.items()
    for variant in variants
}

def normalize_soft_skills(raw_skills, model, threshold=0.4):
    """
    1) Try exact lookup in SOFT_SKILL_TAXONOMY.
    2) If no exact variant match, embed via `model.encode(...)`
       and pick the taxonomy key with highest cosine, if above threshold.
    Returns a deduped list of canonical soft‐skill keys.
    """
    normalized = []
    for s in raw_skills:
        key = s.strip().lower()
        # strip everything except letters, numbers, hyphens and spaces
        key = re.sub(r"[^a-z0-9\- ]+", "", key)
        
        canon = VARIANT_TO_CANONICAL_SOFT.get(key)
        if canon:
            normalized.append(canon)
            continue
        
        candidates = list(SOFT_SKILL_TAXONOMY.keys())
        embeddings = model.encode([key] + candidates, convert_to_tensor=True)
        sims = util.cos_sim(embeddings[0:1], embeddings[1:]).squeeze(0)  
        
        best_idx = int(torch.argmax(sims))
        if sims[best_idx] >= threshold:
            normalized.append(candidates[best_idx])
    
    # dedupe while preserving order
    seen = set()
    out = []
    for c in normalized:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


TITLE_SYNONYMS = {
    # General role normalization
    "developer": "engineer",
    "dev":       "engineer",
    "programmer": "engineer",
    
    # Common variations
    "sysadmin":  "system administrator",
    "cto":       "chief technology officer",
    "sre":       "site reliability engineer",
    "qa":        "quality assurance",
    "pm":        "project manager",
    "po":        "product owner",
    "ux":        "user experience",
    "ui":        "user interface",
    "infosec":   "security",
    "ml":        "machine learning",
    "ai":        "artificial intelligence"
}


def normalize_title(text: str) -> str:
    """
    Lowercase, strip punctuation & parentheticals, map common synonyms.
   
    """
    # 1) remove any parenthetical
    text = re.sub(r"\(.*?\)", "", text)
    # 2) lowercase
    text = text.lower()
    # 3) map tokens via TITLE_SYNONYMS
    tokens = text.split()
    tokens = [ TITLE_SYNONYMS.get(tok, tok) for tok in tokens ]
    return " ".join(tokens)