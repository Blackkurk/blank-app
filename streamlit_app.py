import streamlit as st
import json
from typing import List, Set
from PyPDF2 import PdfReader
import spacy
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
import requests

# Load English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def custom_suffix_cleaning(lemma: str) -> str:
    if lemma.endswith("ed") and len(lemma) > 4:
        return lemma[:-2]
    if lemma.endswith("ing") and len(lemma) > 5:
        return lemma[:-3]
    return lemma

def extract_text_from_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def preprocess_text(text: str) -> List[str]:
    doc = nlp(text)
    lemmas = []
    preserve_stop_words = {"and", "to", "of"}
    for token in doc:
        if token.is_alpha:
            lemma = custom_suffix_cleaning(token.lemma_.lower())
            if (token.is_stop and lemma not in preserve_stop_words) or lemma in spacy.lang.en.stop_words.STOP_WORDS:
                continue
            if len(lemma) > 2 or lemma in preserve_stop_words:
                lemmas.append(lemma)
    return lemmas

def generate_ngrams(words: List[str], n: int) -> List[str]:
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def process_resumes(uploaded_files) -> List[dict]:
    results = []
    for uploaded_file in uploaded_files:
        raw_text = extract_text_from_pdf(uploaded_file)
        lemmas = preprocess_text(raw_text)
        bigrams = generate_ngrams(lemmas, 2)
        trigrams = generate_ngrams(lemmas, 3)
        tokenized_phrases = lemmas + bigrams + trigrams
        result = {
            "id": uploaded_file.name.split(".")[0],
            "lemmas": lemmas,
            "tokenized_phrases": tokenized_phrases
        }
        results.append(result)
    return results

def match_skills_with_tokenization(job_data: List[dict], resume_data: List[dict], threshold: int = 80) -> List[dict]:
    results = []
    for job in job_data:
        job_id = job['id']
        job_title = job['title']
        required_skills = set(skill.lower() for skill in job['required_skills'])
        for resume in resume_data:
            resume_id = resume['id']
            tokenized_phrases = set(phrase.lower() for phrase in resume['tokenized_phrases'])
            matched_skills = set()
            unmatched_skills = set()
            for skill in required_skills:
                found = False
                for phrase in tokenized_phrases:
                    if fuzz.ratio(skill, phrase) >= threshold:
                        matched_skills.add(skill)
                        found = True
                        break
                if not found:
                    unmatched_skills.add(skill)
            results.append({
                "job_id": job_id,
                "job_title": job_title,
                "resume_id": resume_id,
                "matched_skills": list(matched_skills),
                "unmatched_skills": list(unmatched_skills)
            })
    return results

def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def calculate_similarity_and_rank(matching_results: list) -> list:
    ranked_results = []
    for result in matching_results:
        job_id = result["job_id"]
        job_title = result["job_title"]
        matched_skills = set(result["matched_skills"])
        unmatched_skills = set(result["unmatched_skills"])
        required_skills = matched_skills | unmatched_skills
        similarity = calculate_jaccard_similarity(matched_skills, required_skills)
        ranked_results.append({
            "job_id": job_id,
            "job_title": job_title,
            "resume_id": result["resume_id"],
            "similarity": round(similarity, 2),
            "matched_skills": list(matched_skills),
            "unmatched_skills": list(unmatched_skills)
        })
    ranked_results.sort(key=lambda x: x["similarity"], reverse=True)
    return ranked_results

def plot_similarity_bar_chart(ranked_results, resume_id):
    filtered = [r for r in ranked_results if r["resume_id"] == resume_id]
    if not filtered:
        st.warning(f"No results for resume: {resume_id}")
        return

    n = 25
    for i, chunk in enumerate([filtered[:n], filtered[n:]]):
        if not chunk:
            continue
        job_titles = [f'{r["job_id"]}: {r["job_title"]}' for r in chunk]
        similarities = [r["similarity"] for r in chunk]
        fig, ax = plt.subplots(figsize=(max(16, len(job_titles)*0.7), 10))
        ax.bar(job_titles, similarities, color='skyblue')
        ax.set_ylabel('Jaccard Similarity')
        ax.set_xlabel('Job Titles')
        ax.set_title(f'Similarity Scores for Resume: {resume_id}')
        plt.xticks(rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_yticks([round(x * 0.05, 2) for x in range(0, 21)])
        plt.tight_layout()
        st.pyplot(fig)

# --- Streamlit UI ---

url = "https://drive.google.com/file/d/1TSvDI4EgXL7OvkBYJLl1Dm7LSnmQB5CT/view?usp=sharing"
response = requests.get(url)
job_data = response.json()

st.set_page_config(page_title="Resume Screening System", layout="wide")
st.markdown("# 🛫 Resume Screening System")
st.markdown("### Upload your CVs and get instant matching & visual analytics!")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Home (Upload & Process)", "View Results"])

if page == "Home (Upload & Process)":
    st.subheader("Step 1: Upload CVs")
    upload_mode = st.radio("Select upload mode:", ["Single CV (PDF)", "Multiple CVs (PDF Folder)"])
    if upload_mode == "Single CV (PDF)":
        uploaded_files = st.file_uploader("Upload a single PDF resume", type="pdf", accept_multiple_files=False)
        if uploaded_files:
            uploaded_files = [uploaded_files]
    else:
        uploaded_files = st.file_uploader("Upload multiple PDF resumes (select all PDFs in a folder)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.info("Processing resumes, please wait...")
        processed_resumes = process_resumes(uploaded_files)
        st.session_state["processed_resumes"] = processed_resumes

        matching_results = match_skills_with_tokenization(job_data, processed_resumes)
        st.session_state["matching_results"] = matching_results

        ranked_results = calculate_similarity_and_rank(matching_results)
        st.session_state["ranked_results"] = ranked_results

        st.success("All steps completed! Go to 'View Results' in the sidebar to see the charts.")
        st.write("### Processed Resumes")
        st.json(processed_resumes)
        st.write("### Matching Results")
        st.json(matching_results)
        st.write("### Ranked Results")
        st.json(ranked_results)

if page == "View Results":
    st.subheader("Similarity Charts")
    if "ranked_results" not in st.session_state:
        st.warning("Please upload and process resumes first on the Home page.")
    else:
        resume_ids = sorted(set(r["resume_id"] for r in st.session_state["ranked_results"]))
        selected_resume = st.selectbox("Select a resume to view similarity chart:", resume_ids)
        plot_similarity_bar_chart(st.session_state["ranked_results"], selected_resume)