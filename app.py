import io
import json
import os
import re
from typing import Any

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types


# -----------------------------
# App configuration
# -----------------------------
MODEL = "gemini-3.6-flash"

st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)


# -----------------------------
# Helper functions
# -----------------------------
def get_api_key() -> str:
    """Get Gemini API key from Streamlit Secrets or environment variable."""
    try:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        return os.getenv("GEMINI_API_KEY", "").strip()


def extract_pdf_text(uploaded_file) -> str:
    """Extract selectable text from every page of a PDF."""
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))

    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()


def analyze_resume(
    resume_text: str,
    job_description: str,
    api_key: str,
) -> dict[str, Any]:
    """Analyze the resume with Gemini and return structured JSON."""

    client = genai.Client(api_key=api_key)

    job_text = (
        job_description.strip()
        if job_description.strip()
        else "No job description was provided. Evaluate general ATS readiness."
    )

    prompt = f"""
You are an expert ATS resume analyzer and professional resume reviewer.

Analyze the candidate's resume and provide a realistic ATS-style assessment.

IMPORTANT:
- If a job description is provided, evaluate keyword and skill alignment
  against that job description.
- If no job description is provided, evaluate general ATS readiness only.
- Do not invent candidate information.
- Do not claim that the score is an official score from a particular ATS.
- The score is an AI-based ATS-readiness estimate.
- Give specific, actionable improvement suggestions.
- Look for ATS-friendly headings, structure, readability, relevant keywords,
  measurable achievements, skills, experience, education, consistency,
  and formatting problems.
- Do not penalize normal professional formatting without a clear reason.

Return ONLY valid JSON. Do not include markdown or code fences.

Use exactly this JSON structure:

{{
  "ats_score": 0,
  "score_breakdown": {{
    "keyword_match": 0,
    "formatting": 0,
    "experience": 0,
    "skills": 0,
    "achievements": 0,
    "education": 0
  }},
  "summary": "",
  "strengths": [],
  "improvements": [],
  "missing_keywords": [],
  "format_issues": [],
  "section_feedback": {{
    "contact_information": "",
    "professional_summary": "",
    "work_experience": "",
    "education": "",
    "skills": "",
    "projects": "",
    "certifications": ""
  }}
}}

Scoring:
- ats_score: integer from 0 to 100.
- keyword_match: integer from 0 to 20.
- formatting: integer from 0 to 20.
- experience: integer from 0 to 20.
- skills: integer from 0 to 15.
- achievements: integer from 0 to 15.
- education: integer from 0 to 10.
- The six breakdown values must add up to ats_score.
- If a category is not applicable, score it fairly based on what can be
  evaluated from the resume rather than inventing information.

JOB DESCRIPTION:
{job_text}

RESUME:
{resume_text}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    result = json.loads(response.text)

    # Defensive normalization.
    score = int(result.get("ats_score", 0))
    result["ats_score"] = max(0, min(100, score))

    breakdown = result.get("score_breakdown", {})
    categories = [
        "keyword_match",
        "formatting",
        "experience",
        "skills",
        "achievements",
        "education",
    ]

    for category in categories:
        value = int(breakdown.get(category, 0))
        breakdown[category] = max(0, value)

    # Make the displayed total consistent with the category breakdown.
    total = sum(breakdown.values())

    if total != result["ats_score"]:
        result["ats_score"] = max(0, min(100, total))

    result["score_breakdown"] = breakdown

    return result


def display_list(items: list[str], empty_message: str) -> None:
    """Display a list safely."""
    if items:
        for item in items:
            st.write(f"• {item}")
    else:
        st.write(empty_message)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown(
        """
### How it works

1. Upload your resume PDF.
2. Optionally paste a job description.
3. Click **Analyze Resume**.
4. Gemini evaluates ATS readiness.
5. Review the score and improvements.

**AI Model:** Gemini 2.5 Flash
"""
    )

    st.warning(
        "Never put your Gemini API key directly in app.py or commit it to GitHub."
    )


# -----------------------------
# Main UI
# -----------------------------
st.title("📄 Resume ATS Analyzer")

st.write(
    "Upload your resume to get an AI-based ATS readiness score, "
    "score breakdown, missing keywords, and actionable improvements."
)

st.info(
    "For the most useful analysis, paste the job description for the position "
    "you are applying to."
)

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"],
    help="Upload a text-based PDF resume.",
)

job_description = st.text_area(
    "Job Description (Optional)",
    height=220,
    placeholder=(
        "Paste the job description here. "
        "The analyzer will compare your resume against it."
    ),
)

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
)


# -----------------------------
# Analysis
# -----------------------------
if analyze_button:

    if resume_file is None:
        st.error("Please upload your resume PDF first.")
        st.stop()

    api_key = get_api_key()

    if not api_key:
        st.error(
            "Gemini API key is missing. Add GEMINI_API_KEY to Streamlit "
            "Secrets or set it as an environment variable."
        )
        st.stop()

    try:
        with st.spinner("Reading your resume..."):
            resume_text = extract_pdf_text(resume_file)

        if not resume_text:
            st.error(
                "No readable text was found in this PDF. "
                "Please upload a text-based PDF rather than a scanned image."
            )
            st.stop()

        # Basic protection against accidentally processing an extremely large file.
        if len(resume_text) > 100_000:
            st.warning(
                "The extracted resume is unusually large. Only the first "
                "100,000 characters will be analyzed."
            )
            resume_text = resume_text[:100_000]

        # Detect nearly empty/scanned PDFs.
        visible_text_length = len(re.sub(r"\s+", "", resume_text))

        if visible_text_length < 100:
            st.error(
                "Very little text could be extracted from this PDF. "
                "Please upload a normal text-based PDF."
            )
            st.stop()

        with st.spinner("Analyzing your resume with Gemini..."):
            result = analyze_resume(
                resume_text=resume_text,
                job_description=job_description,
                api_key=api_key,
            )

        # -----------------------------
        # Score
        # -----------------------------
        score = result["ats_score"]

        st.divider()
        st.subheader("📊 ATS Score")

        score_col1, score_col2 = st.columns([1, 3])

        with score_col1:
            st.metric("ATS Readiness", f"{score}/100")

        with score_col2:
            st.progress(score / 100)

        st.caption(
            "This is an AI-based ATS-readiness estimate, not an official score "
            "from a specific applicant tracking system."
        )

        # -----------------------------
        # Score breakdown
        # -----------------------------
        st.subheader("📈 Score Breakdown")

        breakdown = result.get("score_breakdown", {})

        breakdown_data = [
            ("Keyword Match", breakdown.get("keyword_match", 0), 20),
            ("Formatting", breakdown.get("formatting", 0), 20),
            ("Experience", breakdown.get("experience", 0), 20),
            ("Skills", breakdown.get("skills", 0), 15),
            ("Achievements", breakdown.get("achievements", 0), 15),
            ("Education", breakdown.get("education", 0), 10),
        ]

        cols = st.columns(3)

        for index, (name, value, maximum) in enumerate(breakdown_data):
            with cols[index % 3]:
                st.metric(name, f"{value}/{maximum}")

        # -----------------------------
        # Summary
        # -----------------------------
        st.subheader("📝 Overall Assessment")
        st.write(result.get("summary", "No summary was returned."))

        # -----------------------------
        # Strengths and improvements
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Strengths")
            display_list(
                result.get("strengths", []),
                "No specific strengths were returned.",
            )

        with col2:
            st.subheader("🔧 Improvements")
            display_list(
                result.get("improvements", []),
                "No specific improvements were returned.",
            )

        # -----------------------------
        # Keywords
        # -----------------------------
        st.subheader("🔑 Missing / Useful Keywords")

        keywords = result.get("missing_keywords", [])

        if keywords:
            keyword_text = "  ".join(
                f"`{keyword}`" for keyword in keywords
            )
            st.markdown(keyword_text)
        else:
            st.write("No major missing keywords were identified.")

        # -----------------------------
        # Formatting
        # -----------------------------
        st.subheader("📄 Formatting Issues")

        display_list(
            result.get("format_issues", []),
            "No major formatting issues were identified.",
        )

        # -----------------------------
        # Section feedback
        # -----------------------------
        st.subheader("📚 Section-by-Section Feedback")

        feedback = result.get("section_feedback", {})

        section_names = {
            "contact_information": "Contact Information",
            "professional_summary": "Professional Summary",
            "work_experience": "Work Experience",
            "education": "Education",
            "skills": "Skills",
            "projects": "Projects",
            "certifications": "Certifications",
        }

        for key, title in section_names.items():
            with st.expander(title):
                st.write(
                    feedback.get(
                        key,
                        "No feedback was returned for this section.",
                    )
                )

        # -----------------------------
        # Resume text preview
        # -----------------------------
        with st.expander("🔎 View Extracted Resume Text"):
            st.text(resume_text)

    except json.JSONDecodeError:
        st.error(
            "Gemini returned an unexpected response format. "
            "Please try the analysis again."
        )

    except Exception as exc:
        st.error(
            "Something went wrong while analyzing the resume."
        )
        st.exception(exc)


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Resume ATS Analyzer • Built with Streamlit, Gemini 2.5 Flash, and pypdf"
)
