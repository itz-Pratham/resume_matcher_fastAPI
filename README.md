# Resume Matcher MVP

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

### Using the API with Postman

1. **Upload the job description**

   Send a `POST` request to `/jd/upload` with `form-data` containing a single
   key `file` pointing to a PDF file. If a non-PDF file is uploaded the API will
   return a `400` response explaining that only PDF files are accepted.

2. **Upload resumes**

   Send a `POST` request to `/resumes/upload` with `form-data` key `file` set to
   a ZIP archive of resume PDFs or DOCX files. Invalid ZIP files or other file
   types will result in a clear error message.

3. **Score candidates**

   After uploading, use the returned resume paths and job description embedding
   to `POST` to `/score`. If any path is wrong or a file type is unsupported you
   will receive a detailed `400` error describing how to fix the issue.

   Use this template to enter the raw content in JSON format
   ```ts
   {
   "jd_embedding": [
   ],
   "job_title": ,
   "resume_paths": [
     ]
   }
   ```

