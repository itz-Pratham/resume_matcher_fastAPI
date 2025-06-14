# Resume Matcher FastAPI

This project ranks resumes against a job description using spaCy and Sentence-Transformers.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# python-multipart is needed for FastAPI file uploads (installed via requirements.txt)
pytest  # run unit tests
```

## Running

```bash
streamlit run main.py
```

### FastAPI Backend

```bash
uvicorn app.main:app --reload
```

The API exposes endpoints to upload a job description, upload resumes, compute
screening scores and fetch previous runs. A frontend such as Next.js can call
these endpoints using regular `fetch` requests:

```ts
await fetch('/jd/upload', { method: 'POST', body: formData });
```

The app now provides per-candidate **View Details** navigation and keeps a history
of past job description runs in `uploads/job_history.json` which can be recalled
from the sidebar.
proper navigation of each profile is also present

