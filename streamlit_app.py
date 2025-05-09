import streamlit as st
import os
import json
from typing import List, Set
from PyPDF2 import PdfReader
import spacy

# 加載 NLP 模型
nlp = spacy.load("en_core_web_sm")

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

    # 保留的停用詞
    preserve_stop_words = {"and", "to", "of"}

    for token in doc:
        if token.is_alpha:
            lemma = custom_suffix_cleaning(token.lemma_.lower())

            # 過濾停用詞，但保留關鍵詞
            if (token.is_stop and lemma not in preserve_stop_words) or lemma in spacy.lang.en.stop_words.STOP_WORDS:
                continue

            # 忽略過短的詞
            if len(lemma) > 2 or lemma in preserve_stop_words:
                lemmas.append(lemma)

    return lemmas

def generate_ngrams(words: List[str], n: int) -> List[str]:
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def process_resumes(uploaded_files):
    results = []
    for uploaded_file in uploaded_files:
        # 保存上傳的文件到臨時目錄
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 提取文本並進行預處理
        raw_text = extract_text_from_pdf(uploaded_file.name)
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

        # 刪除臨時文件
        os.remove(uploaded_file.name)
    return results

def match_skills_with_tokenization(job_file: str, resume_data: List[dict]) -> List[dict]:
    """
    根據 Tokenization 功能，將簡歷技能與職缺技能進行匹配
    """
    # 加載職缺數據
    with open(job_file, 'r', encoding='utf-8') as f:
        job_data = json.load(f)

    results = []

    # 遍歷每個職缺
    for job in job_data:
        job_id = job['id']
        job_title = job['title']
        required_skills = set(skill.lower() for skill in job['required_skills'])  # 忽略大小寫

        # 遍歷每份簡歷
        for resume in resume_data:
            resume_id = resume['id']
            tokenized_phrases = set(phrase.lower() for phrase in resume['tokenized_phrases'])  # 忽略大小寫

            # 匹配技能
            matched_skills = required_skills & tokenized_phrases  # 交集
            unmatched_skills = required_skills - matched_skills  # 差集

            # 保存結果
            results.append({
                "job_id": job_id,
                "job_title": job_title,
                "resume_id": resume_id,
                "matched_skills": list(matched_skills),
                "unmatched_skills": list(unmatched_skills)
            })

    return results

def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    計算 Jaccard 相似度
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def calculate_similarity_and_rank(input_file: str, output_file: str):
    """
    計算 Jaccard 相似度，按職缺排序並輸出到 JSON 文件
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ranked_results = []

    for result in data:
        job_id = result["job_id"]
        job_title = result["job_title"]
        matched_skills = set(result["matched_skills"])
        unmatched_skills = set(result["unmatched_skills"])
        required_skills = matched_skills | unmatched_skills

        # 計算 Jaccard 相似度
        similarity = calculate_jaccard_similarity(matched_skills, required_skills)

        ranked_results.append({
            "job_id": job_id,
            "job_title": job_title,
            "resume_id": result["resume_id"],
            "similarity": round(similarity, 2),
            "matched_skills": list(matched_skills),
            "unmatched_skills": list(unmatched_skills)
        })

    # 按相似度排序
    ranked_results.sort(key=lambda x: x["similarity"], reverse=True)

    # 保存結果到 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ranked_results, f, ensure_ascii=False, indent=2)

    return ranked_results

# Streamlit GUI
st.title("Resume Screening System")

st.header("Step 1: Upload CVs")
uploaded_files = st.file_uploader("Upload PDF resumes", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.header("Step 2: Process Resumes")
    if st.button("Run Preprocessing"):
        # 處理簡歷
        processed_resumes = process_resumes(uploaded_files)

        # 保存結果到 JSON 文件
        output_path = "/Users/stella/Desktop/Resume_Screening_Solution/processed_resumes.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_resumes, f, indent=2, ensure_ascii=False)

        # 使用 session_state 保存處理結果
        st.session_state["processed_resumes"] = processed_resumes

        st.success(f"Processed {len(processed_resumes)} resumes and saved to {output_path}")

        # 顯示結果
        st.header("Step 3: Processed Results")
        st.json(processed_resumes)

# 匹配技能
if "processed_resumes" in st.session_state:
    st.header("Step 4: Match Skills")
    if st.button("Match Skills"):
        job_file = "/Users/stella/Desktop/Resume_Screening_Solution/job_parsed.json"
        matching_results = match_skills_with_tokenization(job_file, st.session_state["processed_resumes"])

        # 保存匹配結果到 JSON 文件
        matching_output_path = "/Users/stella/Desktop/Resume_Screening_Solution/matching_results.json"
        with open(matching_output_path, "w", encoding="utf-8") as f:
            json.dump(matching_results, f, indent=2, ensure_ascii=False)

        # 使用 session_state 保存匹配結果
        st.session_state["matching_results"] = matching_results

        st.success(f"Skill matching completed and saved to {matching_output_path}")

        # 顯示匹配結果
        st.header("Matching Results")
        st.json(matching_results)

# 計算相似度
if "matching_results" in st.session_state:
    st.header("Step 5: Calculate Similarity")
    if st.button("Calculate Similarity"):
        ranked_results = calculate_similarity_and_rank(
            input_file="/Users/stella/Desktop/Resume_Screening_Solution/matching_results.json",
            output_file="/Users/stella/Desktop/Resume_Screening_Solution/ranked_results.json"
        )

        st.success("Similarity calculation completed and saved to ranked_results.json")

        # 顯示相似度結果
        st.header("Similarity Results")
        st.json(ranked_results)