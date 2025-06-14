import pandas as pd
from utils.parse_utils import parse_resume_fields, parse_jd_fields, encode_text, compute_similarity_score

def test_score_rounding(tmp_path):
    resume_text = "John Doe\nSkills: python machine learning\nEducation: BSc Computer Science"
    jd_text = "Python machine learning skills needed. BSc Computer Science required"

    resume = parse_resume_fields(resume_text)
    jd_info = parse_jd_fields(jd_text)
    jd_emb = encode_text(jd_info["required_skills"])
    score, ats, skill, edu = compute_similarity_score(resume, jd_emb)

    df = pd.DataFrame([
        {
            "Candidate Name": resume["name"],
            "Suggested Role": jd_info["job_title"],
            "Total Score": round(score, 2),
            "ATS Score": round(ats, 2),
            "Skill Match Score": round(skill, 2),
            "Education Score": round(edu, 2),
        }
    ])
    csv_path = tmp_path / "out.csv"
    df.to_csv(csv_path, index=False)
    saved = pd.read_csv(csv_path)
    assert saved.iloc[0]["Total Score"] == round(score, 2)
