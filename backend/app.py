"""
app.py - DetectAI web server (FastAPI).

  POST /api/predict     {"text": "..."}  -> verdict, probability, highlights, sentence scores, style
  GET  /api/health      -> is the server up, which model is loaded
  GET  /api/model-info  -> model description, thresholds and test-set metrics
  GET  /                -> the website (frontend/ folder)

Start:  venv\\Scripts\\python -m uvicorn backend.app:app --port 8000   (or double-click run.bat)
"""

import sys
import time
from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Make sure the NLTK data we need is present (first run on a new laptop)
for resource, pkg in [("tokenizers/punkt_tab", "punkt_tab"), ("corpora/stopwords", "stopwords"),
                      ("corpora/wordnet", "wordnet"),
                      ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(pkg, quiet=True)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from backend.detector import Detector  # noqa: E402

MIN_WORDS = 5
MAX_WORDS = 3000

app = FastAPI(title="DetectAI", description="Explainable detection of AI-generated text")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

detector = Detector()   # loads BERT if its folder exists, otherwise the baseline


class PredictRequest(BaseModel):
    text: str = ""


def error(message, status=400):
    return JSONResponse(status_code=status, content={"error": message})


@app.post("/api/predict")
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    n_words = len(text.split())
    if n_words == 0:
        return error("Please paste or type some text to analyse.")
    if n_words < MIN_WORDS:
        return error(f"Please enter at least {MIN_WORDS} words (you entered {n_words}).")
    if n_words > MAX_WORDS:
        return error(f"Text is too long ({n_words} words). Please paste at most {MAX_WORDS} words.")
    t0 = time.time()
    result = detector.predict(text)
    result["seconds"] = round(time.time() - t0, 2)
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "model": detector.kind, "model_name": detector.model_name}


@app.get("/api/model-info")
def model_info():
    return detector.info()


# The website itself (mounted last so the /api routes take priority)
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
