import spacy
import subprocess

model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except OSError:
    # 如果模型不存在，则自动下载
    subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
    nlp = spacy.load(model_name)
