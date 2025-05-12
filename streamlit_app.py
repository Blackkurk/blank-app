import streamlit as st
import os
import re
import json
from typing import List, Set
from PyPDF2 import PdfReader
import spacy
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import requests

nlp = spacy.load("en_core_web_sm")

# Define a dictionary for section keywords
SECTION_KEYWORDS = {
    "education": [
        "education", "qualification", "academic background", "academic qualifications"
    ],
    "experience": [
        "working experience", "employment", "internship", "work history", "professional experience"
    ],
    "skills": [
        "skills", "key competencies", "technical skills", "competencies"
    ]
}

# Define a dictionary for degree variants
DEGREE_LEVEL = {
    "high school diploma": 1,
    "associate": 2,
    "bachelor": 3,
    "postgraduate diploma": 4,
    "master": 5,
    "mba": 6,
    "phd": 7
}

def degree_level_score(required, found):
    if not required or not found:
        return 0
    req = DEGREE_LEVEL.get(required.lower())
    fnd = DEGREE_LEVEL.get(found.lower())
    if req is None or fnd is None:
        return 0
    if fnd >= req:
        return 100
    else:
        return int(100 * fnd / req)

def custom_suffix_cleaning(lemma: str) -> str:
    if lemma.endswith("ed") and len(lemma) > 4:
        return lemma[:-2]
    if lemma.endswith("ing") and len(lemma) > 5:
        return lemma[:-3]
    return lemma

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")
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

def extract_education(text):
    edu_pattern = re.compile(
        r'(?i)(.*?)\b(higher diploma|bachelor|master|phd|associate|degree|high school diploma|certificate)\b.*?(\d{4})\s*(?:–|-|to)?\s*(\d{4}|present)?',
        re.DOTALL)
    matches = edu_pattern.findall(text)
    results = []
    for match in matches:
        institution = match[0].strip().replace("\n", " ")
        degree = match[1]
        start_year = match[2]
        end_year = match[3] or "present"
        results.append({
            "institution": institution,
            "degree": degree,
            "start_year": start_year,
            "end_year": end_year
        })
    return results

def extract_work_experience(text):
    work_section = re.search(r'(?i)(Working Experience|Work Experience|Employment)(.*?)(Awards|Skills|Technical|Language|$)', text, re.DOTALL)
    experiences = []
    if work_section:
        content = work_section.group(2).strip()
        job_blocks = re.split(r'\n{2,}', content)
        for block in job_blocks:
            lines = block.strip().split('\n')
            if not lines:
                continue
            full_block = ' '.join(lines).strip()
            title_line = lines[0].strip() if lines else ""
            desc = ' '.join(lines[1:]).strip() if len(lines) > 1 else ""
            combined = full_block
            year_match = re.search(r'(\d{2}\.\d{4}|\d{4})\s*(?:–|-|to)\s*(\d{2}\.\d{4}|\d{4}|present)', combined)
            if year_match:
                start = year_match.group(1)
                end = year_match.group(2)
                try:
                    if '.' in start:
                        start_year = int(start.split('.')[1])
                    else:
                        start_year = int(start)
                    if '.' in end:
                        end_year = int(end.split('.')[1])
                    else:
                        end_year = int(end) if end.isdigit() else 2025
                except:
                    start_year = ""
                    end_year = "present"
            else:
                start_year = ""
                end_year = "present"
            experiences.append({
                "title": title_line,
                "description": desc,
                "start_year": str(start_year),
                "end_year": str(end_year)
            })
    return experiences

def extract_year_ranges(work_info):
    ranges = []
    year_range_pattern = re.compile(r'(19|20)\d{2}\s*(?:–|-|to)\s*(\d{4}|present)', re.IGNORECASE)
    for job in work_info:
        texts = [job.get("title", ""), job.get("description", "")]
        for text in texts:
            for match in year_range_pattern.finditer(text):
                start = int(match.group(0)[:4])
                end_raw = match.group(2).lower()
                end = int(end_raw) if end_raw.isdigit() else 2025
                if start <= end:
                    ranges.append((start, end))
    return merge_year_ranges(ranges)

def merge_year_ranges(ranges):
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def calculate_years_experience(work_info):
    ranges = extract_year_ranges(work_info)
    total = sum(end - start + 1 for start, end in ranges)
    return total

def classify_segments(text):
    doc = nlp(text)
    edu_keywords = {"university", "college", "diploma", "degree", "education", "institute", "school"}
    work_keywords = {"company", "engineer", "intern", "developer", "assistant", "analyst", "manager", "helper"}
    education_blocks = []
    work_blocks = []
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    for para in paragraphs:
        token_set = set([token.lemma_.lower() for token in nlp(para)])
        if token_set & edu_keywords:
            education_blocks.append(para)
        if token_set & work_keywords:
            work_blocks.append(para)
    return "\n\n".join(education_blocks), "\n\n".join(work_blocks)

def process_resumes(uploaded_files):
    results = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        raw_text = extract_text_from_pdf(uploaded_file.name)
        edu_text, work_text = classify_segments(raw_text)
        education_info = extract_education(edu_text)
        work_info = extract_work_experience(work_text)
        years_experience = calculate_years_experience(work_info)
        degree = education_info[0]["degree"] if education_info else ""
        lemmas = preprocess_text(raw_text)
        bigrams = generate_ngrams(lemmas, 2)
        trigrams = generate_ngrams(lemmas, 3)
        tokenized_phrases = lemmas + bigrams + trigrams
        result = {
            "id": uploaded_file.name.split(".")[0],
            "lemmas": lemmas,
            "tokenized_phrases": tokenized_phrases,
            "degree": degree,
            "years_experience": years_experience,
            "education_info": education_info,
            "work_info": work_info
        }
        results.append(result)
        os.remove(uploaded_file.name)
    return results
def match_skills_with_tokenization(job_data: List[dict], resume_data: List[dict], threshold: int = 80) -> List[dict]:
    results = []
    for job in job_data:
        job_id = job['id']
        job_title = job['title']
        required_skills = set(skill.lower() for skill in job['required_skills'])
        required_degree = (job.get("degree_required") or "").strip().lower()
        required_exp = job.get("experience_years_required", None)
        for resume in resume_data:
            resume_id = resume['id']
            tokenized_phrases = set(phrase.lower() for phrase in resume['tokenized_phrases'])
            lemmas = resume.get("lemmas", [])
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
            resume_degree = (resume.get("degree") or "").strip().lower()
            degree_score = degree_level_score(required_degree, resume_degree)
            resume_exp = resume.get("years_experience")
            if resume_exp is not None and required_exp is not None:
                exp_score = 100 if resume_exp >= required_exp else int(100 * resume_exp / required_exp)
            else:
                exp_score = 0
            results.append({
                "job_id": job_id,
                "job_title": job_title,
                "resume_id": resume_id,
                "matched_skills": list(matched_skills),
                "unmatched_skills": list(unmatched_skills),
                "degree_required": required_degree,
                "degree_found": resume_degree,
                "degree_score": degree_score,
                "experience_required": required_exp,
                "experience_found": resume_exp,
                "exp_score": exp_score
            })
    return results

def calculate_final_score(skill_score, exp_score, degree_score):
    SKILL_WEIGHT = 0.5
    EXP_WEIGHT = 0.3
    DEGREE_WEIGHT = 0.2
    return (
        skill_score * SKILL_WEIGHT +
        (exp_score / 100) * EXP_WEIGHT +
        (degree_score / 100) * DEGREE_WEIGHT
    )

def mark_high_potential(skill_score, exp_score, degree_score):
    return {
        "high_potential_skill": skill_score > 0.85,
        "high_potential_experience": exp_score >= 100,
        "high_potential_degree": degree_score > 90
    }

def calculate_similarity_and_rank(matching_results: list) -> list:
    ranked_results = []
    for result in matching_results:
        matched_skills = set(result.get("matched_skills", []))
        unmatched_skills = set(result.get("unmatched_skills", []))
        required_skills = matched_skills | unmatched_skills
        skill_score = len(matched_skills) / len(required_skills) if required_skills else 0
        exp_score = result.get("exp_score", 0)
        degree_score = result.get("degree_score", 0)
        final_score = calculate_final_score(skill_score, exp_score, degree_score)
        high_potential = mark_high_potential(skill_score, exp_score, degree_score)
        ranked_results.append({
            "job_id": result["job_id"],
            "job_title": result["job_title"],
            "resume_id": result["resume_id"],
            "final_score": round(final_score, 4),
            "skill_score": round(skill_score, 4),
            "exp_score": exp_score,
            "degree_score": degree_score,
            **high_potential,
            "matched_skills": list(matched_skills),
            "unmatched_skills": list(unmatched_skills),
            "degree_required": result.get("degree_required"),
            "degree_found": result.get("degree_found"),
            "experience_required": result.get("experience_required"),
            "experience_found": result.get("experience_found")
        })
    ranked_results.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked_results

def plot_grouped_bar_chart(ranked_results, resume_id=None, job_id=None, page_num=1, page_size=25):
    mode = st.radio("View Mode", ["By Resume", "By Job"], horizontal=True, key="bar_mode")
    if mode == "By Resume":
        filtered = [r for r in ranked_results if r["resume_id"] == resume_id]
        filtered = sorted(filtered, key=lambda x: x["final_score"], reverse=True)
        label_list = [f'{r["job_id"]}: {r["job_title"]}' for r in filtered]
        chart_title = f'Scores for Resume: {resume_id}'

        score_options = {
            "Final Score": "final_score",
            "Skill Score": "skill_score",
            "Experience Score": "exp_score",
            "Degree Score": "degree_score"
        }
        selected_score_label = st.selectbox("Select score type to display:", list(score_options.keys()), key="score_type_select")
        selected_score_key = score_options[selected_score_label]

        start = (page_num - 1) * page_size
        end = start + page_size
        group = filtered[start:end]
        if not group:
            st.warning("No data in this page.")
            return

        values = []
        for r in group:
            v = r[selected_score_key]
            if selected_score_key in ("exp_score", "degree_score"):
                v = v / 100
            values.append(v)

        x = np.arange(len(group))
        fig, ax = plt.subplots(figsize=(max(24, len(group)*0.7), 12))
        ax.bar(x, values, width=0.6, color="skyblue")
        ax.set_ylabel(selected_score_label)
        ax.set_xlabel('Job Titles')
        ax.set_title(f'{selected_score_label} - {chart_title} (Page {page_num})')
        ax.set_xticks(x)
        ax.set_xticklabels(label_list[start:end], rotation=45, ha='right')
        ax.set_ylim(0, 1.25)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        filtered = [r for r in ranked_results if r["job_id"] == job_id]
        filtered = sorted(filtered, key=lambda x: x["final_score"], reverse=True)
        label_list = [r["resume_id"] for r in filtered]
        chart_title = f'Scores for Job: {job_id}'

        start = (page_num - 1) * page_size
        end = start + page_size
        group = filtered[start:end]
        if not group:
            st.warning("No data in this page.")
            return

        x = np.arange(len(group))
        width = 0.18
        fig, ax = plt.subplots(figsize=(max(24, len(group)*0.7), 12))
        ax.bar(x - 1.5*width, [r["final_score"] for r in group], width, label="Final Score")
        ax.bar(x - 0.5*width, [r["skill_score"] for r in group], width, label="Skill Score")
        ax.bar(x + 0.5*width, [r["exp_score"]/100 for r in group], width, label="Experience Score")
        ax.bar(x + 1.5*width, [r["degree_score"]/100 for r in group], width, label="Degree Score")
        ax.set_ylabel('Score')
        ax.set_xlabel('Resume IDs')
        ax.set_title(f'All Scores - {chart_title} (Page {page_num})')
        ax.set_xticks(x)
        ax.set_xticklabels(label_list[start:end], rotation=45, ha='right')
        ax.set_ylim(0, 1.25)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # 分頁導航
    total = len(filtered)
    total_pages = (total + page_size - 1) // page_size
    st.write(f"Page {page_num} of {total_pages}")
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if page_num > 1:
            if st.button("Previous Page", key="prev_bar"):
                st.session_state['bar_chart_page'] = page_num - 1
    with col3:
        if page_num < total_pages:
            if st.button("Next Page", key="next_bar"):
                st.session_state['bar_chart_page'] = page_num + 1

def plot_heatmap(ranked_results, resume_id=None, job_id=None, page_num=1, page_size=25):
    mode = st.radio("View Mode", ["By Resume", "By Job"], horizontal=True, key="heatmap_mode")
    if mode == "By Resume":
        filtered = [r for r in ranked_results if r["resume_id"] == resume_id]
        if not filtered:
            st.warning(f"No results for resume: {resume_id}")
            return
        total = len(filtered)
        start = (page_num - 1) * page_size
        end = start + page_size
        group = filtered[start:end]
        if not group:
            st.warning("No data in this page.")
            return
        data = np.array([
            [r["final_score"], r["skill_score"], r["exp_score"]/100, r["degree_score"]/100]
            for r in group
        ])
        job_titles = [f'{r["job_id"]}: {r["job_title"]}' for r in group]
        categories = ["Final", "Skill", "Exp", "Degree"]
        fig, ax = plt.subplots(figsize=(max(18, len(job_titles)*0.7), 8))
        sns.heatmap(data.T, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
                    xticklabels=job_titles, yticklabels=categories, ax=ax)
        ax.set_title(f'Heatmap of Scores for Resume: {resume_id} (Page {page_num})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        # 分頁導航
        total_pages = (total + page_size - 1) // page_size
        st.write(f"Page {page_num} of {total_pages}")
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if page_num > 1:
                if st.button("Previous Page", key="prev_heatmap"):
                    st.session_state['heatmap_page'] = page_num - 1
        with col3:
            if page_num < total_pages:
                if st.button("Next Page", key="next_heatmap"):
                    st.session_state['heatmap_page'] = page_num + 1
    else:
        filtered = [r for r in ranked_results if r["job_id"] == job_id]
        if not filtered:
            st.warning(f"No results for job: {job_id}")
            return
        total = len(filtered)
        start = (page_num - 1) * page_size
        end = start + page_size
        group = filtered[start:end]
        if not group:
            st.warning("No data in this page.")
            return
        data = np.array([
            [r["final_score"], r["skill_score"], r["exp_score"]/100, r["degree_score"]/100]
            for r in group
        ])
        resume_ids = [r["resume_id"] for r in group]
        categories = ["Final", "Skill", "Exp", "Degree"]
        fig, ax = plt.subplots(figsize=(max(18, len(resume_ids)*0.7), 8))
        sns.heatmap(data.T, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True,
                    xticklabels=resume_ids, yticklabels=categories, ax=ax)
        ax.set_title(f'Heatmap of Scores for Job: {job_id} (Page {page_num})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        # 分頁導航
        total_pages = (total + page_size - 1) // page_size
        st.write(f"Page {page_num} of {total_pages}")
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if page_num > 1:
                if st.button("Previous Page", key="prev_heatmap"):
                    st.session_state['heatmap_page'] = page_num - 1
        with col3:
            if page_num < total_pages:
                if st.button("Next Page", key="next_heatmap"):
                    st.session_state['heatmap_page'] = page_num + 1

def show_high_potential_applicants(ranked_results):
    st.subheader("High Potential Applicants")
    df = [
        r for r in ranked_results
        if r.get("exp_score", 0) == 100
        and r.get("degree_score", 0) == 100
        and r.get("skill_score", 0) > 0
        and r.get("exp_score", 0) > 0
        and r.get("degree_score", 0) > 0
    ]
    if not df:
        st.info("No high potential applicants found.")
        return
    st.write(f"Total: {len(df)}")
    st.dataframe(df)

def show_applicant_overview(ranked_results):
    st.subheader("Applicant Overview")
    resume_ids = sorted(set(r["resume_id"] for r in ranked_results))
    selected_resume = st.selectbox("Select an applicant (resume):", resume_ids)
    filtered = [r for r in ranked_results if r["resume_id"] == selected_resume]
    filtered = sorted(filtered, key=lambda x: x.get("final_score", x.get("similarity", 0)), reverse=True)
    if not filtered:
        st.info("No results for this applicant.")
        return
    st.write(f"Total jobs matched: {len(filtered)}")
    st.dataframe(filtered)

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Screening System", layout="wide")
st.markdown("# 🛫 Resume Screening System")
st.markdown("### Upload your CVs and get instant matching & visual analytics!")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", [
        "Home (Upload & Process)",
        "Grouped Bar Chart",
        "Heatmap",
        "High Potential Applicants",
        "Applicant Overview"
    ])

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

        url = "https://drive.google.com/uc?export=download&id=1RF3vet0zAQbk1u47YzoXUlgyUpVOx2wW"
        response = requests.get(url)
        job_data = response.json()["jobs"]
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
        
if page == "Grouped Bar Chart":
    resume_ids = sorted(set(r["resume_id"] for r in st.session_state["ranked_results"]))
    job_ids = sorted(set(r["job_id"] for r in st.session_state["ranked_results"]))
    selected_resume = st.selectbox("Select a resume:", resume_ids)
    selected_job = st.selectbox("Select a job:", job_ids)
    if 'bar_chart_page' not in st.session_state:
        st.session_state['bar_chart_page'] = 1
    plot_grouped_bar_chart(
        st.session_state["ranked_results"],
        resume_id=selected_resume,
        job_id=selected_job,
        page_num=st.session_state['bar_chart_page'],
        page_size=25
    )
        
if page == "Heatmap":
    if "ranked_results" not in st.session_state:
        st.warning("Please upload and process resumes first on the Home page.")
    else:
        resume_ids = sorted(set(r["resume_id"] for r in st.session_state["ranked_results"]))
        job_ids = sorted(set(r["job_id"] for r in st.session_state["ranked_results"]))
        selected_resume = st.selectbox("Select a resume:", resume_ids)
        selected_job = st.selectbox("Select a job:", job_ids)
        if 'heatmap_page' not in st.session_state:
            st.session_state['heatmap_page'] = 1
        plot_heatmap(
            st.session_state["ranked_results"],
            resume_id=selected_resume,
            job_id=selected_job,
            page_num=st.session_state['heatmap_page'],
            page_size=25
        )

if page == "High Potential Applicants":
    if "ranked_results" not in st.session_state:
        st.warning("Please upload and process resumes first on the Home page.")
    else:
        show_high_potential_applicants(st.session_state["ranked_results"])

if page == "Applicant Overview":
    if "ranked_results" not in st.session_state:
        st.warning("Please upload and process resumes first on the Home page.")
    else:
        show_applicant_overview(st.session_state["ranked_results"])