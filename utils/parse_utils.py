import re
import spacy
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

@lru_cache(maxsize=1)
def get_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_spacy_model():
    return spacy.load("en_core_web_sm")

model = get_transformer()
nlp = get_spacy_model()


def extract_name(text: str) -> str:
    """Heuristic extraction of the candidate name from resume text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header = "\n".join(lines[:5])
    doc = nlp(header)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    if lines:
        first = lines[0]
        if len(first.split()) <= 4 and not re.search(r"\d", first.lower()):
            return first
    return "Unknown"

def parse_resume_fields(text: str) -> dict:
    doc = nlp(text)
    name = extract_name(text)
    skills, education, experience = [], [], []

    # Simple keyword-based segmentation
    for sent in doc.sents:
        s = sent.text.lower()
        if 'skill' in s:
            skills.append(sent.text)
        if 'education' in s or 'bachelor' in s or 'degree' in s:
            education.append(sent.text)
        if 'experience' in s or 'worked at' in s or 'years' in s:
            experience.append(sent.text)

    return {
        "name": name,
        "skills": ' '.join(skills),
        "education": ' '.join(education),
        "experience": ' '.join(experience),
    }

def parse_jd_fields(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines else "Unknown"
    required_skills = []
    for line in lines:
        low = line.lower()
        if re.search(r"requirement|qualification|skill", low):
            required_skills.append(line)
        elif re.match(r"^[\-*•]", line):
            required_skills.append(line)

    return {
        "job_title": title,
        "required_skills": ' '.join(required_skills)
    }

def encode_text(text):
    return model.encode(text, convert_to_tensor=True)


def compute_similarity_score(resume, jd_embedding):
    """Compute similarity between a resume and a pre-computed JD embedding."""
    resume_text = resume["skills"] + " " + resume["education"] + " " + resume["experience"]
    embeddings = model.encode([resume_text, resume["skills"], resume["education"]], convert_to_tensor=True)

    total_score = util.pytorch_cos_sim(embeddings[0], jd_embedding).item()
    skill_score = util.pytorch_cos_sim(embeddings[1], jd_embedding).item()
    edu_score = util.pytorch_cos_sim(embeddings[2], jd_embedding).item()

    ats_score = 0.5 * skill_score + 0.5 * edu_score
    final_score = 0.5 * ats_score + 0.3 * skill_score + 0.2 * edu_score

    return final_score, ats_score, skill_score, edu_score