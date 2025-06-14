import streamlit as st

# st.set_page_config must be the first Streamlit command executed. It is
# placed immediately after importing Streamlit to avoid the "set_page_config
# can only be called once" error when other modules use Streamlit decorators
# during import.
st.set_page_config(page_title="Resume Matcher MVP", layout="wide")

from utils.file_utils import (
    save_uploaded_file,
    unzip_resumes,
    extract_text_from_pdf,
    extract_text_from_docx,
)
from utils.parse_utils import (
    parse_resume_fields,
    parse_jd_fields,
    compute_similarity_score,
    encode_text,
)
import os
import json
import pandas as pd
from datetime import datetime

st.title("📄 AI-Powered Resume Screening MVP")

def do_rerun():
    """Compatibility wrapper for Streamlit rerun"""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def render_results(results, key_prefix=""):
    """Render candidate results as a table with a per-row view button."""
    df = pd.DataFrame(results).sort_values(by="Total Score", ascending=False)
    display_cols = [
        "Candidate Name",
        "Suggested Role",
        "Total Score",
        "ATS Score",
        "Skill Match Score",
        "Education Score",
        "Match Priority",
    ]
    table = df[display_cols]
    widths = [3, 3, 2, 2, 2, 2, 2, 1]
    header = st.columns(widths)
    for col, name in zip(header[:-1], display_cols):
        col.markdown(f"**{name}**")
    header[-1].markdown("**View Details**")

    for i, row in table.iterrows():
        cols = st.columns(widths)
        for c, val in zip(cols[:-1], row):
            c.write(val)
        if cols[-1].button("🔍", key=f"{key_prefix}view_{i}"):
            st.session_state.current_results = results
            st.session_state.current_index = i
            st.session_state.page = "details"
            do_rerun()

    st.download_button(
        "Download CSV",
        table.to_csv(index=False),
        file_name="ranked_candidates.csv",
        key=f"{key_prefix}download",
    )

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


if "page" not in st.session_state:
    st.session_state.page = "upload"
if "current_results" not in st.session_state:
    st.session_state.current_results = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_zip = st.file_uploader("Upload Resumes (ZIP of PDFs or DOCX)", type=["zip"])

if st.button("Run Screening"):
    if jd_file and resume_zip:
        jd_path = save_uploaded_file(jd_file, "uploads/jd.pdf")
        zip_path = save_uploaded_file(resume_zip, "uploads/resumes.zip")
        resume_paths = unzip_resumes(zip_path, "uploads/resumes")

        jd_text = extract_text_from_pdf(jd_path)
        jd_info = parse_jd_fields(jd_text)
        jd_embedding = encode_text(jd_info["required_skills"])

        st.subheader("📋 Job Description Info")
        st.json(jd_info)

        results = []
        st.subheader("👥 Candidate Rankings")
        for path in resume_paths:
            lower = path.lower()
            if lower.endswith(".pdf"):
                text = extract_text_from_pdf(path)
            elif lower.endswith(".docx"):
                text = extract_text_from_docx(path)
            else:
                continue

            parsed = parse_resume_fields(text)
            score, ats_score, skill_score, edu_score = compute_similarity_score(parsed, jd_embedding)

            results.append({
                "Candidate Name": parsed['name'],
                "Suggested Role": jd_info['job_title'],
                "Total Score": round(score, 2),
                "ATS Score": round(ats_score, 2),
                "Skill Match Score": round(skill_score, 2),
                "Education Score": round(edu_score, 2),
                "Match Priority": "High" if score >= 0.75 else ("Medium" if score >= 0.5 else "Low"),
                "skills": parsed['skills'],
                "education": parsed['education'],
                "experience": parsed['experience'],
                "path": path
            })

        render_results(results)

        history = load_history()
        history.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "jd_text": jd_text, "results": results})
        save_history(history)
        st.session_state.current_results = results
    else:
        st.warning("Please upload both JD and resumes.")

history = load_history()
selected = st.sidebar.selectbox("📜 Past JD runs", [""] + [h["timestamp"] for h in history])
if selected:
    idx = next(i for i,h in enumerate(history) if h["timestamp"] == selected)
    st.session_state.current_results = history[idx]["results"]
    st.session_state.page = "list"

if st.session_state.page == "details" and st.session_state.current_results:
    rec = st.session_state.current_results[st.session_state.current_index]
    st.subheader(rec["Candidate Name"])
    st.markdown("### Skills")
    st.write(rec["skills"])
    st.markdown("### Experience")
    st.write(rec["experience"])
    st.markdown("### Education")
    st.write(rec["education"])
    with open(rec["path"], "rb") as f:
        st.download_button(
            "Open Resume",
            f,
            file_name=os.path.basename(rec["path"]),
            key=f"resume_{st.session_state.current_index}",
        )
    cols = st.columns(3)
    if cols[0].button("← Prev", disabled=st.session_state.current_index==0):
        st.session_state.current_index -= 1
        do_rerun()
    if cols[1].button("Back to list"):
        st.session_state.page = "list"
        do_rerun()
    if cols[2].button("Next →", disabled=st.session_state.current_index==len(st.session_state.current_results)-1):
        st.session_state.current_index += 1
        do_rerun()
elif st.session_state.current_results:
    render_results(st.session_state.current_results, key_prefix="saved_")
