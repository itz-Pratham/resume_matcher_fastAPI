from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import os
import json
from datetime import datetime

from utils.file_utils import (
    save_uploaded_file,
    unzip_resumes,
    extract_text_from_pdf,
    extract_text_from_docx,
)
from utils.parse_utils import (
    parse_jd_fields,
    parse_resume_fields,
    encode_text,
    compute_similarity_score,
)

from .models import JDInfo, ResumeUploadResponse, ScoreRequest, ScoreResponse, CandidateResult

app = FastAPI(title="Resume Screening API")


@app.get("/")
async def root():
    """Basic health check at the API root."""
    return {"message": "Resume Screening API is running"}

HISTORY_PATH = "uploads/job_history.json"


def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


@app.post("/jd/upload", response_model=JDInfo)
async def upload_job_description(file: UploadFile = File(...)):
    """Upload a job description in PDF format."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a file ending in .pdf",
        )

    path = save_uploaded_file(file, "uploads/jd.pdf")

    try:
        text = extract_text_from_pdf(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    info = parse_jd_fields(text)
    embedding = encode_text(info["required_skills"]).tolist()
    return JDInfo(
        job_title=info["job_title"],
        required_skills=info["required_skills"],
        embedding=embedding,
    )


@app.post("/resumes/upload", response_model=ResumeUploadResponse)
async def upload_resumes(file: UploadFile = File(...)):
    """Upload a ZIP of resumes and return the extracted file paths."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a ZIP archive containing resume files (.pdf or .docx).",
        )

    zip_path = save_uploaded_file(file, "uploads/resumes.zip")

    try:
        paths = unzip_resumes(zip_path, "uploads/resumes")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ResumeUploadResponse(resume_paths=paths)


@app.post("/score", response_model=ScoreResponse)
async def score_candidates(request: ScoreRequest):
    """Compute candidate scores given a JD embedding and resume paths."""
    jd_embedding = request.jd_embedding
    results = []
    for path in request.resume_paths:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=400,
                detail=f"Resume file not found: {path}. Ensure the path is correct.",
            )

        lower = path.lower()
        try:
            if lower.endswith(".pdf"):
                text = extract_text_from_pdf(path)
            elif lower.endswith(".docx"):
                text = extract_text_from_docx(path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type for {path}. Use PDF or DOCX resumes.",
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        parsed = parse_resume_fields(text)
        score, ats, skill, edu = compute_similarity_score(parsed, jd_embedding)
        results.append(
            {
                "candidate_name": parsed["name"],
                "suggested_role": request.job_title,
                "total_score": round(score, 2),
                "ats_score": round(ats, 2),
                "skill_match_score": round(skill, 2),
                "education_score": round(edu, 2),
                "match_priority": "High"
                if score >= 0.75
                else ("Medium" if score >= 0.5 else "Low"),
                "skills": parsed["skills"],
                "education": parsed["education"],
                "experience": parsed["experience"],
                "path": path,
            }
        )

    history = load_history()
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "jd_title": request.job_title,
            "results": results,
        }
    )
    save_history(history)

    return ScoreResponse(results=[CandidateResult(**r) for r in results])


@app.get("/history")
async def get_history():
    """Retrieve history of past screenings."""
    return load_history()

