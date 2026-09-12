import hashlib
import os
from pathlib import Path
from huggingface_hub import HfApi

folder = Path('models/v10')
checkpoint = folder / 'best_cer_v10.pt'
expected = (folder / 'checkpoint_sha256.txt').read_text().split()[0]
actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError('Le checkpoint ne correspond pas à son empreinte SHA256.')
token = os.environ.get('HF_TOKEN')
if not token:
    raise RuntimeError('Configurer le secret GitHub Actions HF_TOKEN avec accès en écriture au dépôt Hugging Face.')
repo_id = 'IntelligenceResearchLab/Hausa-OCR-AJAMI'
result = HfApi(token=token).upload_folder(
    repo_id=repo_id,
    repo_type='model',
    folder_path=folder,
    ignore_patterns=['__pycache__/*', '*.pyc'],
    commit_message='Publish trained V10 from GitHub ' + os.environ.get('GITHUB_SHA', 'local')[:12],
)
print('Publication terminée :', result.commit_url)
summary = os.environ.get('GITHUB_STEP_SUMMARY')
if summary:
    with open(summary, 'a', encoding='utf-8') as handle:
        handle.write(f'Modèle V10 publié : {result.commit_url}\n\nSHA256 : `{actual}`\n')
