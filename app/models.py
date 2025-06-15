from pydantic import BaseModel
from typing import List

class JDInfo(BaseModel):
    job_title: str
    required_skills: str
    embedding: List[float]

class ResumeUploadResponse(BaseModel):
    resume_paths: List[str]

class ScoreRequest(BaseModel):
    jd_embedding: List[float]
    job_title: str
    resume_paths: List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "jd_embedding": [0.1, 0.2, 0.3],
                "job_title": "Data Scientist",
                "resume_paths": ["uploads/resumes/john.pdf"],
            }
        }
    }

class CandidateResult(BaseModel):
    candidate_name: str
    suggested_role: str
    total_score: float
    ats_score: float
    skill_match_score: float
    education_score: float
    match_priority: str
    path: str
    skills: str
    education: str
    experience: str

class ScoreResponse(BaseModel):
    results: List[CandidateResult]

    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "candidate_name": "John Doe",
                        "suggested_role": "Data Scientist",
                        "total_score": 0.92,
                        "ats_score": 0.88,
                        "skill_match_score": 0.9,
                        "education_score": 0.8,
                        "match_priority": "High",
                        "path": "uploads/resumes/john.pdf",
                        "skills": "python machine learning",
                        "education": "BSc Computer Science",
                        "experience": "3 years at XYZ",
                    }
                ]
            }
        }
    }

