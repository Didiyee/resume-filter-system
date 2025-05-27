import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import re

# --- Prepare page---
st.set_page_config(page_title="Resume Filter System", layout="centered")
# st.title("📄 Resume Filter System")
logo = Image.open("logo.png")

col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=130)
with col2:
    st.title(" Resume Filter System 📄 \n")
# --- Load model BERT ---
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_bert_model()

# --- Extract text from PDF ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- Extract Experience from cv --
def extract_experience_section(cv_text):
    exp_match = re.search(r"(Experience|EXPERIENCE|Work History|Professional Experience)(.*?)(Education|EDUCATION|Skills|SKILLS|$)", cv_text, re.DOTALL)
    return exp_match.group(2).strip() if exp_match else "Not found."

# --- Extract job requirments --
def extract_job_requirements(jd_text):
    lines = jd_text.split("\n")
    keywords = [line.strip() for line in lines if any(word in line.lower() for word in ["require", "must", "should", "responsible", "experience"])]
    return "\n".join(keywords[:5]) or "Not clearly stated."

# --- CV Analyse ---
def analyze_resumes(jd_text, cv_files):
    results = []
    jd_clean = jd_text.strip()
    job_requirements = extract_job_requirements(jd_clean)
    jd_embedding = model.encode(jd_clean, convert_to_tensor=True)

    for cv in cv_files:
        cv_text = extract_text_from_pdf(cv)
        experience = extract_experience_section(cv_text)

        cv_embedding = model.encode(cv_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(jd_embedding, cv_embedding).item()

        results.append({
            "name": cv.name,
            "score": score,
            "experience": experience,
            "job_requirements": job_requirements
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# --- Main App---
def main():
    # logo = Image.open("Logo.png")
    # st.image(logo, width=200)
    st.header("1. Upload Job Description")
    jd_file = st.file_uploader("Upload Job Description (txt, pdf)", type=["txt", "pdf"])

    st.header("2. Upload Candidate CVs")
    cv_files = st.file_uploader("Upload CVs (pdf only)", type=["pdf"], accept_multiple_files=True)

    if st.button("🚀 Analyze Resumes", key="analyze_button"):
        if not jd_file:
            st.warning("⚠️ Please upload a Job Description file.")
        elif not cv_files:
            st.warning("⚠️ Please upload at least one Candidate CV.")
        else:
            st.info("🔍 Processing... Please wait...")

            jd_text = extract_text_from_pdf(jd_file)
            results = analyze_resumes(jd_text, cv_files)

            st.header("🏆 Analysis Results")
            for i , result in enumerate(results):
                st.subheader(f"Rank {i+1}: {result['name']}")
                st.markdown(f"**Score:** `{result['score']:.2f}`")
                with st.expander("📄 Match Details"):
                    st.markdown("**📝 Job Requirements (Extracted):**")
                    st.code(result["job_requirements"], language="text")
                    st.markdown("**👤 Candidate Experience (Extracted):**")
                    st.code(result["experience"], language="text")

if __name__ == "__main__":
    main()